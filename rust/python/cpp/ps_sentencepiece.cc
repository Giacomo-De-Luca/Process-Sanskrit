#include "ps_sentencepiece.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include "sentencepiece_processor.h"

struct PsSentencePiece {
  sentencepiece::SentencePieceProcessor processor;
};

namespace {

char *copy_error(const std::string &message) {
  auto *copy = static_cast<char *>(std::malloc(message.size() + 1));
  if (copy == nullptr) {
    return nullptr;
  }
  std::memcpy(copy, message.data(), message.size());
  copy[message.size()] = '\0';
  return copy;
}

PsSentencePiecePieces failed(const std::string &message) {
  return PsSentencePiecePieces{nullptr, 0, nullptr, 0, 1, copy_error(message)};
}

}  // namespace

extern "C" PsSentencePiece *ps_sentencepiece_create(const char *model_data,
                                                     std::size_t model_len,
                                                     char **error) {
  if (error != nullptr) {
    *error = nullptr;
  }
  try {
    if (model_data == nullptr || model_len == 0) {
      if (error != nullptr) {
        *error = copy_error("SentencePiece model data is empty");
      }
      return nullptr;
    }
    auto *processor = new PsSentencePiece();
    const auto status = processor->processor.LoadFromSerializedProto(
        absl::string_view(model_data, model_len));
    if (!status.ok()) {
      if (error != nullptr) {
        *error = copy_error(status.ToString());
      }
      delete processor;
      return nullptr;
    }
    return processor;
  } catch (const std::exception &exception) {
    if (error != nullptr) {
      *error = copy_error(exception.what());
    }
    return nullptr;
  } catch (...) {
    if (error != nullptr) {
      *error = copy_error("unknown exception while loading SentencePiece");
    }
    return nullptr;
  }
}

extern "C" void ps_sentencepiece_destroy(PsSentencePiece *processor) {
  delete processor;
}

extern "C" PsSentencePiecePieces ps_sentencepiece_encode(
    const PsSentencePiece *processor, const char *text, std::size_t text_len) {
  try {
    if (processor == nullptr || text == nullptr) {
      return failed("SentencePiece processor or input is null");
    }
    std::vector<std::string> pieces;
    const auto status =
        processor->processor.Encode(std::string(text, text_len), &pieces);
    if (!status.ok()) {
      return failed(status.ToString());
    }
    if (pieces.empty()) {
      return PsSentencePiecePieces{nullptr, 0, nullptr, 0, 0, nullptr};
    }
    std::size_t data_len = 0;
    for (const auto &piece : pieces) {
      data_len += piece.size();
    }
    auto *offsets = static_cast<std::size_t *>(
        std::malloc((pieces.size() + 1) * sizeof(std::size_t)));
    if (offsets == nullptr) {
      return failed("out of memory while copying SentencePiece offsets");
    }
    // Allocate one byte for the theoretically possible all-empty result so
    // pointer arithmetic below never operates on null.
    auto *data = static_cast<char *>(std::malloc(data_len == 0 ? 1 : data_len));
    if (data == nullptr) {
      std::free(offsets);
      return failed("out of memory while copying SentencePiece strings");
    }
    offsets[0] = 0;
    std::size_t cursor = 0;
    for (std::size_t index = 0; index < pieces.size(); ++index) {
      if (!pieces[index].empty()) {
        std::memcpy(data + cursor, pieces[index].data(), pieces[index].size());
      }
      cursor += pieces[index].size();
      offsets[index + 1] = cursor;
    }
    return PsSentencePiecePieces{data, data_len, offsets, pieces.size(), 0,
                                 nullptr};
  } catch (const std::exception &exception) {
    return failed(exception.what());
  } catch (...) {
    return failed("unknown exception while encoding with SentencePiece");
  }
}

extern "C" void ps_sentencepiece_pieces_destroy(PsSentencePiecePieces result) {
  std::free(result.data);
  std::free(result.offsets);
  std::free(result.error);
}

extern "C" void ps_sentencepiece_error_destroy(char *error) {
  std::free(error);
}
