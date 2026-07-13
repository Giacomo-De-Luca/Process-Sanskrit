use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use fst::{Map, Streamer};

use crate::CoreError;

const VARIANT_MAGIC: &[u8; 8] = b"PSSV0001";
const VARIANT_VERSION: u32 = 1;
const VARIANT_HEADER_LEN: usize = 24;

/// One inverse sandhi rule. `left` may start with `^` to require index zero.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RuleVariant {
    pub left: String,
    pub right: String,
}

impl RuleVariant {
    pub fn new(left: impl Into<String>, right: impl Into<String>) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
        }
    }
}

enum RuleStore {
    Memory(BTreeMap<String, Vec<RuleVariant>>),
    Native(NativeRules),
}

struct NativeRules {
    afters: Map<Vec<u8>>,
    variants: Vec<u8>,
    group_offsets: Vec<usize>,
    after_len_max: usize,
}

/// Indexed inverse sandhi rules.
pub struct SandhiRules {
    store: RuleStore,
    after_len_max: usize,
}

impl std::fmt::Debug for SandhiRules {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SandhiRules")
            .field("after_len_max", &self.after_len_max)
            .finish_non_exhaustive()
    }
}

impl SandhiRules {
    /// Construct an in-memory rule index, primarily for small tests.
    pub fn from_memory(mut rules: BTreeMap<String, Vec<RuleVariant>>) -> Self {
        for variants in rules.values_mut() {
            variants.sort();
            variants.dedup();
        }
        let after_len_max = rules.keys().map(String::len).max().unwrap_or(0);
        Self {
            store: RuleStore::Memory(rules),
            after_len_max,
        }
    }

    /// Load the deterministic FST and flat variant table emitted by the
    /// resource builder.
    pub fn load(after_fst: &Path, variants_bin: &Path) -> Result<Self, CoreError> {
        let afters = fs::read(after_fst).map_err(|error| CoreError::io(after_fst, error))?;
        let variants =
            fs::read(variants_bin).map_err(|error| CoreError::io(variants_bin, error))?;
        Self::from_owned_bytes(afters, variants)
    }

    pub(crate) fn from_owned_bytes(afters: Vec<u8>, variants: Vec<u8>) -> Result<Self, CoreError> {
        let afters = Map::new(afters)
            .map_err(|error| CoreError::asset("sandhi_after.fst", error.to_string()))?;
        let (group_offsets, declared_variant_count) = parse_variant_header(&variants)?;

        let mut after_len_max = 0;
        let mut expected_group = 0_u64;
        let mut stream = afters.stream();
        while let Some((after, group)) = stream.next() {
            if after.is_empty() || !after.is_ascii() {
                return Err(CoreError::asset(
                    "sandhi_after.fst",
                    "sandhi keys must be non-empty canonical ASCII SLP1",
                ));
            }
            if group != expected_group {
                return Err(CoreError::asset(
                    "sandhi_after.fst",
                    format!("group id {group} appears where {expected_group} was expected"),
                ));
            }
            expected_group += 1;
            after_len_max = after_len_max.max(after.len());
        }
        if expected_group as usize + 1 != group_offsets.len() {
            return Err(CoreError::asset(
                "sandhi_variants.bin",
                "group count disagrees with sandhi_after.fst",
            ));
        }

        let native = NativeRules {
            afters,
            variants,
            group_offsets,
            after_len_max,
        };
        let actual_variant_count = native.validate_records()?;
        if actual_variant_count != declared_variant_count {
            return Err(CoreError::asset(
                "sandhi_variants.bin",
                format!(
                    "declares {declared_variant_count} variants but contains {actual_variant_count}"
                ),
            ));
        }

        Ok(Self {
            after_len_max: native.after_len_max,
            store: RuleStore::Native(native),
        })
    }

    /// Expand all inverse rules at each byte position in canonical SLP1.
    ///
    /// `stop=None` and `stop=Some(0)` both mirror Python's `stop or len(word)`.
    pub fn split_all(
        &self,
        word: &str,
        start: usize,
        stop: Option<usize>,
    ) -> Result<BTreeSet<(String, String)>, CoreError> {
        if !word.is_ascii() {
            return Err(CoreError::InvalidSlp1(word.to_owned()));
        }
        let stop = match stop {
            Some(0) | None => word.len(),
            Some(stop) => stop.min(word.len()),
        };
        let mut splits = BTreeSet::new();
        for index in start.min(word.len())..stop {
            let remaining = word.len() - index;
            for after_len in 1..=self.after_len_max.min(remaining) {
                let after = &word[index..index + after_len];
                self.for_each_variant(after, |left_rule, right_rule| {
                    let left_rule = match left_rule.strip_prefix('^') {
                        Some(left) if index == 0 => left,
                        Some(_) => return,
                        None => left_rule,
                    };
                    let mut left = String::with_capacity(index + left_rule.len());
                    left.push_str(&word[..index]);
                    left.push_str(left_rule);

                    let suffix = &word[index + after_len..];
                    let mut right = String::with_capacity(right_rule.len() + suffix.len());
                    right.push_str(right_rule);
                    right.push_str(suffix);
                    splits.insert((left, right));
                })?;
            }
        }
        Ok(splits)
    }

    fn for_each_variant(
        &self,
        after: &str,
        mut visit: impl FnMut(&str, &str),
    ) -> Result<(), CoreError> {
        match &self.store {
            RuleStore::Memory(rules) => {
                if let Some(variants) = rules.get(after) {
                    for variant in variants {
                        visit(&variant.left, &variant.right);
                    }
                }
                Ok(())
            }
            RuleStore::Native(native) => {
                let Some(group) = native.afters.get(after) else {
                    return Ok(());
                };
                native.for_each_variant(group as usize, visit)
            }
        }
    }
}

impl NativeRules {
    fn validate_records(&self) -> Result<u64, CoreError> {
        let mut count = 0_u64;
        for group in 0..self.group_offsets.len().saturating_sub(1) {
            let mut invalid = false;
            self.for_each_variant(group, |left, right| {
                count += 1;
                invalid |= left.is_empty() || !left.is_ascii() || !right.is_ascii();
            })?;
            if invalid {
                return Err(CoreError::asset(
                    "sandhi_variants.bin",
                    format!("rule group {group} contains an empty left side or non-ASCII SLP1"),
                ));
            }
        }
        Ok(count)
    }

    fn for_each_variant(
        &self,
        group: usize,
        mut visit: impl FnMut(&str, &str),
    ) -> Result<(), CoreError> {
        let Some((&start, &end)) = self
            .group_offsets
            .get(group)
            .zip(self.group_offsets.get(group + 1))
        else {
            return Err(CoreError::asset(
                "sandhi_variants.bin",
                format!("rule group {group} is out of range"),
            ));
        };
        let mut cursor = start;
        while cursor < end {
            let left_len = read_u32(&self.variants, &mut cursor, end)? as usize;
            let right_len = read_u32(&self.variants, &mut cursor, end)? as usize;
            let left = read_utf8(&self.variants, &mut cursor, end, left_len)?;
            let right = read_utf8(&self.variants, &mut cursor, end, right_len)?;
            visit(left, right);
        }
        if cursor != end {
            return Err(CoreError::asset(
                "sandhi_variants.bin",
                format!("rule group {group} ends in the middle of a record"),
            ));
        }
        Ok(())
    }
}

fn parse_variant_header(bytes: &[u8]) -> Result<(Vec<usize>, u64), CoreError> {
    if bytes.len() < VARIANT_HEADER_LEN || &bytes[..8] != VARIANT_MAGIC {
        return Err(CoreError::asset(
            "sandhi_variants.bin",
            "missing PSSV0001 header",
        ));
    }
    let mut cursor = 8;
    let version = read_u32(bytes, &mut cursor, bytes.len())?;
    if version != VARIANT_VERSION {
        return Err(CoreError::asset(
            "sandhi_variants.bin",
            format!("unsupported format version {version}"),
        ));
    }
    let group_count = read_u32(bytes, &mut cursor, bytes.len())? as usize;
    let variant_count = read_u64(bytes, &mut cursor, bytes.len())?;
    let offset_count = group_count
        .checked_add(1)
        .ok_or_else(|| CoreError::asset("sandhi_variants.bin", "group table length overflow"))?;
    let records_start = VARIANT_HEADER_LEN
        .checked_add(offset_count.checked_mul(8).ok_or_else(|| {
            CoreError::asset("sandhi_variants.bin", "group table length overflow")
        })?)
        .ok_or_else(|| CoreError::asset("sandhi_variants.bin", "header length overflow"))?;
    if records_start > bytes.len() {
        return Err(CoreError::asset(
            "sandhi_variants.bin",
            "group offset table exceeds file length",
        ));
    }

    let mut offsets = Vec::with_capacity(offset_count);
    for _ in 0..offset_count {
        let offset = read_u64(bytes, &mut cursor, bytes.len())?;
        let offset = usize::try_from(offset).map_err(|_| {
            CoreError::asset("sandhi_variants.bin", "group offset exceeds address space")
        })?;
        offsets.push(offset);
    }
    if offsets.first().copied() != Some(records_start)
        || offsets.last().copied() != Some(bytes.len())
        || offsets.windows(2).any(|window| window[0] > window[1])
    {
        return Err(CoreError::asset(
            "sandhi_variants.bin",
            "invalid or non-monotonic group offsets",
        ));
    }
    Ok((offsets, variant_count))
}

fn read_u32(bytes: &[u8], cursor: &mut usize, end: usize) -> Result<u32, CoreError> {
    let raw = read_exact::<4>(bytes, cursor, end)?;
    Ok(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], cursor: &mut usize, end: usize) -> Result<u64, CoreError> {
    let raw = read_exact::<8>(bytes, cursor, end)?;
    Ok(u64::from_le_bytes(raw))
}

fn read_exact<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
    end: usize,
) -> Result<[u8; N], CoreError> {
    let next = cursor
        .checked_add(N)
        .ok_or_else(|| CoreError::asset("sandhi_variants.bin", "record offset overflow"))?;
    if next > end || next > bytes.len() {
        return Err(CoreError::asset(
            "sandhi_variants.bin",
            "unexpected end of file",
        ));
    }
    let raw = bytes[*cursor..next]
        .try_into()
        .expect("slice length is checked");
    *cursor = next;
    Ok(raw)
}

fn read_utf8<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    end: usize,
    len: usize,
) -> Result<&'a str, CoreError> {
    let next = cursor
        .checked_add(len)
        .ok_or_else(|| CoreError::asset("sandhi_variants.bin", "record length overflow"))?;
    if next > end || next > bytes.len() {
        return Err(CoreError::asset(
            "sandhi_variants.bin",
            "unexpected end of variant record",
        ));
    }
    let value = std::str::from_utf8(&bytes[*cursor..next]).map_err(|error| {
        CoreError::asset("sandhi_variants.bin", format!("invalid UTF-8: {error}"))
    })?;
    *cursor = next;
    Ok(value)
}
