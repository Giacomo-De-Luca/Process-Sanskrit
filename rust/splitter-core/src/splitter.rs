use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use crate::{CoreError, Resources, SequenceScorer};

type NodeId = usize;
const MAX_GRAPH_DEPTH: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitOptions {
    pub limit: usize,
    pub scored: bool,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            scored: true,
        }
    }
}

/// Thread-safe splitter; all mutable graph state is request-local.
#[derive(Clone, Debug)]
pub struct Splitter {
    resources: Arc<Resources>,
}

impl Splitter {
    pub fn new(resources: Arc<Resources>) -> Self {
        Self { resources }
    }

    /// Split canonical SLP1, returning `None` when no valid graph root exists.
    pub fn split(
        &self,
        slp1: &str,
        options: SplitOptions,
    ) -> Result<Option<Vec<Vec<String>>>, CoreError> {
        if !slp1.is_ascii() {
            return Err(CoreError::InvalidSlp1(slp1.to_owned()));
        }
        let graph = GraphBuilder::new(&self.resources).build(slp1)?;
        if graph.roots.is_empty() {
            return Ok(None);
        }
        if options.scored && options.limit == 0 {
            // Python scores the graph before `islice(..., 0)` raises. Preserve
            // that failure precedence so a broken tokenizer/model is never
            // hidden behind the public zero-limit ValueError.
            EdgeWeights::score(&graph, self.resources.scorer()?)?;
            return Err(CoreError::InvalidLimit { limit: 0 });
        }
        if options.limit == 0 {
            return Ok(Some(Vec::new()));
        }

        let paths = if options.limit > 1000 {
            // Python calls score_graph() before choosing all_simple_paths for
            // this sentinel. The weights do not affect its length ordering,
            // but scorer/tokenizer failures must still remain fatal.
            if options.scored {
                EdgeWeights::score(&graph, self.resources.scorer()?)?;
            }
            let mut paths = graph.all_paths()?;
            paths.sort_by(|left, right| {
                left.nodes
                    .len()
                    .cmp(&right.nodes.len())
                    .then_with(|| graph.canonical_cmp(left, right))
            });
            paths.dedup_by(|right, left| graph.same_tokens(left, right));
            paths
        } else if options.scored {
            let scorer = self.resources.scorer()?;
            let weights = EdgeWeights::score(&graph, scorer)?;
            let mut paths = graph.top_paths(options.limit, &weights)?;
            let sequences = graph.borrowed_sequences(&paths);
            let scores = scorer.score_sequences(&sequences)?;
            if scores.len() != paths.len() {
                return Err(CoreError::ScorerUnavailable(
                    "scorer returned the wrong result count".to_owned(),
                ));
            }
            for (path, score) in paths.iter_mut().zip(scores) {
                if !score.is_finite() {
                    return Err(CoreError::ScorerUnavailable(
                        "scorer returned a non-finite value".to_owned(),
                    ));
                }
                path.full_score = score;
            }
            // Stable sorting deliberately preserves the edge-shortlist order
            // for exact full-sequence score ties.
            paths.sort_by(|left, right| right.full_score.total_cmp(&left.full_score));
            paths
        } else {
            graph.top_paths(options.limit, &EdgeWeights::unweighted(&graph))?
        };

        Ok(Some(
            paths
                .iter()
                .map(|path| graph.owned_sequence(path))
                .collect(),
        ))
    }

    pub fn valid_slp1(&self, word: &str) -> Result<bool, CoreError> {
        if !word.is_ascii() {
            return Err(CoreError::InvalidSlp1(word.to_owned()));
        }
        Ok(self.resources.valid(word))
    }

    pub fn score_slp1(&self, sequences: &[Vec<String>]) -> Result<Vec<f32>, CoreError> {
        if let Some(token) = sequences.iter().flatten().find(|token| !token.is_ascii()) {
            return Err(CoreError::InvalidSlp1(token.clone()));
        }
        let borrowed = sequences
            .iter()
            .map(|sequence| sequence.iter().map(String::as_str).collect())
            .collect::<Vec<Vec<&str>>>();
        self.resources.scorer()?.score_sequences(&borrowed)
    }
}

#[derive(Debug)]
struct Node {
    token: String,
    edges: BTreeSet<NodeId>,
    terminal: bool,
}

#[derive(Debug)]
struct Graph {
    nodes: Vec<Node>,
    roots: Vec<NodeId>,
}

impl Graph {
    fn top_paths(
        &self,
        limit: usize,
        weights: &EdgeWeights,
    ) -> Result<Vec<PathCandidate>, CoreError> {
        let mut cache = HashMap::<NodeId, Vec<PathCandidate>>::new();
        let mut visiting = HashSet::new();
        let mut candidates = Vec::new();
        for &root in &self.roots {
            for mut path in
                self.top_paths_from(root, limit, weights, &mut cache, &mut visiting, 0)?
            {
                path.edge_cost += weights.start[root];
                candidates.push(path);
            }
        }
        candidates.sort_by(|left, right| {
            left.edge_cost
                .total_cmp(&right.edge_cost)
                .then_with(|| self.canonical_cmp(left, right))
        });
        candidates.dedup_by(|right, left| self.same_tokens(left, right));
        candidates.truncate(limit);
        Ok(candidates)
    }

    fn top_paths_from(
        &self,
        node_id: NodeId,
        limit: usize,
        weights: &EdgeWeights,
        cache: &mut HashMap<NodeId, Vec<PathCandidate>>,
        visiting: &mut HashSet<NodeId>,
        depth: usize,
    ) -> Result<Vec<PathCandidate>, CoreError> {
        if depth >= MAX_GRAPH_DEPTH {
            return Err(CoreError::GraphDepthLimit {
                max_depth: MAX_GRAPH_DEPTH,
            });
        }
        if let Some(paths) = cache.get(&node_id) {
            return Ok(paths.clone());
        }
        if !visiting.insert(node_id) {
            return Err(CoreError::GraphCycle(self.nodes[node_id].token.clone()));
        }
        let node = &self.nodes[node_id];
        let mut paths = Vec::new();
        if node.terminal {
            paths.push(PathCandidate {
                nodes: vec![node_id],
                edge_cost: weights.end[node_id],
                full_score: 0.0,
            });
        }
        for &child in &node.edges {
            let suffixes =
                self.top_paths_from(child, limit, weights, cache, visiting, depth + 1)?;
            for suffix in suffixes {
                let mut nodes = Vec::with_capacity(suffix.nodes.len() + 1);
                nodes.push(node_id);
                nodes.extend_from_slice(&suffix.nodes);
                paths.push(PathCandidate {
                    nodes,
                    edge_cost: weights.internal[&(node_id, child)] + suffix.edge_cost,
                    full_score: 0.0,
                });
            }
        }
        visiting.remove(&node_id);
        paths.sort_by(|left, right| {
            left.edge_cost
                .total_cmp(&right.edge_cost)
                .then_with(|| self.canonical_cmp(left, right))
        });
        paths.dedup_by(|right, left| self.same_tokens(left, right));
        paths.truncate(limit);
        cache.insert(node_id, paths.clone());
        Ok(paths)
    }

    fn all_paths(&self) -> Result<Vec<PathCandidate>, CoreError> {
        let mut paths = Vec::new();
        let mut active = HashSet::new();
        let mut prefix = Vec::new();
        for &root in &self.roots {
            self.enumerate_from(root, &mut prefix, &mut active, &mut paths, 0)?;
        }
        Ok(paths)
    }

    fn enumerate_from(
        &self,
        node_id: NodeId,
        prefix: &mut Vec<NodeId>,
        active: &mut HashSet<NodeId>,
        paths: &mut Vec<PathCandidate>,
        depth: usize,
    ) -> Result<(), CoreError> {
        if depth >= MAX_GRAPH_DEPTH {
            return Err(CoreError::GraphDepthLimit {
                max_depth: MAX_GRAPH_DEPTH,
            });
        }
        if !active.insert(node_id) {
            return Err(CoreError::GraphCycle(self.nodes[node_id].token.clone()));
        }
        prefix.push(node_id);
        let node = &self.nodes[node_id];
        if node.terminal {
            paths.push(PathCandidate {
                nodes: prefix.clone(),
                edge_cost: prefix.len() as f64,
                full_score: 0.0,
            });
        }
        for &child in &node.edges {
            self.enumerate_from(child, prefix, active, paths, depth + 1)?;
        }
        prefix.pop();
        active.remove(&node_id);
        Ok(())
    }

    fn borrowed_sequences<'a>(&'a self, paths: &[PathCandidate]) -> Vec<Vec<&'a str>> {
        paths
            .iter()
            .map(|path| {
                path.nodes
                    .iter()
                    .map(|&node| self.nodes[node].token.as_str())
                    .collect()
            })
            .collect()
    }

    fn owned_sequence(&self, path: &PathCandidate) -> Vec<String> {
        path.nodes
            .iter()
            .map(|&node| self.nodes[node].token.clone())
            .collect()
    }

    fn canonical_cmp(&self, left: &PathCandidate, right: &PathCandidate) -> Ordering {
        let token_order = left
            .nodes
            .iter()
            .map(|&node| self.nodes[node].token.as_str())
            .cmp(
                right
                    .nodes
                    .iter()
                    .map(|&node| self.nodes[node].token.as_str()),
            );
        token_order.then_with(|| left.nodes.cmp(&right.nodes))
    }

    fn same_tokens(&self, left: &PathCandidate, right: &PathCandidate) -> bool {
        left.nodes.len() == right.nodes.len()
            && left
                .nodes
                .iter()
                .zip(&right.nodes)
                .all(|(&left, &right)| self.nodes[left].token == self.nodes[right].token)
    }
}

#[derive(Clone, Debug)]
struct PathCandidate {
    nodes: Vec<NodeId>,
    edge_cost: f64,
    full_score: f32,
}

struct GraphBuilder<'a> {
    resources: &'a Resources,
    graph: Graph,
    memo: HashMap<String, Vec<NodeId>>,
    visiting: HashSet<String>,
    validity: HashMap<String, bool>,
}

impl<'a> GraphBuilder<'a> {
    fn new(resources: &'a Resources) -> Self {
        Self {
            resources,
            graph: Graph {
                nodes: Vec::new(),
                roots: Vec::new(),
            },
            memo: HashMap::new(),
            visiting: HashSet::new(),
            validity: HashMap::new(),
        }
    }

    fn build(mut self, slp1: &str) -> Result<Graph, CoreError> {
        self.graph.roots = self.possible_splits(slp1, 0)?;
        self.graph.roots.sort_unstable();
        self.graph.roots.dedup();
        Ok(self.graph)
    }

    fn possible_splits(&mut self, remaining: &str, depth: usize) -> Result<Vec<NodeId>, CoreError> {
        if depth >= MAX_GRAPH_DEPTH {
            return Err(CoreError::GraphDepthLimit {
                max_depth: MAX_GRAPH_DEPTH,
            });
        }
        if let Some(roots) = self.memo.get(remaining) {
            return Ok(roots.clone());
        }
        if !self.visiting.insert(remaining.to_owned()) {
            return Err(CoreError::GraphCycle(remaining.to_owned()));
        }

        let first_space = remaining.find(' ');
        let splits = self
            .resources
            .rules()
            .split_all(remaining, 0, first_space)?;
        let mut roots = BTreeSet::new();
        let mut node_cache = HashMap::<String, NodeId>::new();

        for (left, right) in splits {
            let valid = *self
                .validity
                .entry(left.clone())
                .or_insert_with(|| self.resources.valid(&left));
            if !valid {
                continue;
            }

            if right.is_empty() {
                let node = self.node_for(&left, &mut node_cache);
                self.graph.nodes[node].terminal = true;
                roots.insert(node);
                continue;
            }

            let right_roots = self.possible_splits(right.trim(), depth + 1)?;
            if right_roots.is_empty() {
                continue;
            }
            let node = self.node_for(&left, &mut node_cache);
            self.graph.nodes[node].edges.extend(right_roots);
            roots.insert(node);
        }

        self.visiting.remove(remaining);
        let roots = roots.into_iter().collect::<Vec<_>>();
        self.memo.insert(remaining.to_owned(), roots.clone());
        Ok(roots)
    }

    fn node_for(&mut self, token: &str, cache: &mut HashMap<String, NodeId>) -> NodeId {
        if let Some(&node) = cache.get(token) {
            return node;
        }
        let node = self.graph.nodes.len();
        self.graph.nodes.push(Node {
            token: token.to_owned(),
            edges: BTreeSet::new(),
            terminal: false,
        });
        cache.insert(token.to_owned(), node);
        node
    }
}

struct EdgeWeights {
    start: Vec<f64>,
    end: Vec<f64>,
    internal: HashMap<(NodeId, NodeId), f64>,
}

impl EdgeWeights {
    fn unweighted(graph: &Graph) -> Self {
        let internal = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(node, data)| data.edges.iter().map(move |&child| ((node, child), 1.0)))
            .collect();
        Self {
            start: vec![1.0; graph.nodes.len()],
            end: vec![1.0; graph.nodes.len()],
            internal,
        }
    }

    fn score(graph: &Graph, scorer: &dyn SequenceScorer) -> Result<Self, CoreError> {
        enum EdgeKey {
            Start(NodeId),
            End(NodeId),
            Internal(NodeId, NodeId),
        }

        let mut keys = Vec::new();
        let mut sequences = Vec::new();
        for &root in &graph.roots {
            keys.push(EdgeKey::Start(root));
            sequences.push(vec![graph.nodes[root].token.as_str()]);
        }
        for (node, data) in graph.nodes.iter().enumerate() {
            if data.terminal {
                keys.push(EdgeKey::End(node));
                sequences.push(vec![data.token.as_str()]);
            }
            for &child in &data.edges {
                keys.push(EdgeKey::Internal(node, child));
                sequences.push(vec![data.token.as_str(), graph.nodes[child].token.as_str()]);
            }
        }

        let scores = scorer.score_sequences(&sequences)?;
        if scores.len() != keys.len() {
            return Err(CoreError::ScorerUnavailable(
                "scorer returned the wrong edge count".to_owned(),
            ));
        }
        let mut weights = Self {
            start: vec![f64::INFINITY; graph.nodes.len()],
            end: vec![f64::INFINITY; graph.nodes.len()],
            internal: HashMap::new(),
        };
        for (key, score) in keys.into_iter().zip(scores) {
            if !score.is_finite() {
                return Err(CoreError::ScorerUnavailable(
                    "scorer returned a non-finite edge score".to_owned(),
                ));
            }
            let weight = -(score as f64);
            match key {
                EdgeKey::Start(node) => weights.start[node] = weight,
                EdgeKey::End(node) => weights.end[node] = weight,
                EdgeKey::Internal(left, right) => {
                    weights.internal.insert((left, right), weight);
                }
            }
        }
        Ok(weights)
    }
}
