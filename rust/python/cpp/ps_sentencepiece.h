#ifndef PROCESS_SANSKRIT_SENTENCEPIECE_H_
#define PROCESS_SANSKRIT_SENTENCEPIECE_H_

#include <cstddef>

extern "C" {

struct PsSentencePiece;

struct PsSentencePiecePieces {
  char *data;
  std::size_t data_len;
  std::size_t *offsets;
  std::size_t len;
  int status;
  char *error;
};

PsSentencePiece *ps_sentencepiece_create(const char *model_data,
                                         std::size_t model_len, char **error);
void ps_sentencepiece_destroy(PsSentencePiece *processor);

PsSentencePiecePieces ps_sentencepiece_encode(const PsSentencePiece *processor,
                                               const char *text,
                                               std::size_t text_len);
void ps_sentencepiece_pieces_destroy(PsSentencePiecePieces result);
void ps_sentencepiece_error_destroy(char *error);

}  // extern "C"

#endif  // PROCESS_SANSKRIT_SENTENCEPIECE_H_
