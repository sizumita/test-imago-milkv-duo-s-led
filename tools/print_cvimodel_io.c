#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cviruntime.h>

static void print_dimensions(const CVI_SHAPE *shape) {
  putchar('[');
  for (size_t i = 0; i < shape->dim_size; ++i) {
    if (i != 0) {
      fputs(", ", stdout);
    }
    printf("%d", shape->dim[i]);
  }
  putchar(']');
}

static int read_file(const char *path, uint8_t **bytes_out, size_t *len_out) {
  FILE *fp = fopen(path, "rb");
  long file_size;
  size_t read_len;
  uint8_t *bytes;

  if (fp == NULL) {
    fprintf(stderr, "failed to open %s\n", path);
    return 1;
  }

  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    fprintf(stderr, "failed to seek %s\n", path);
    return 1;
  }
  file_size = ftell(fp);
  if (file_size < 0) {
    fclose(fp);
    fprintf(stderr, "failed to stat %s\n", path);
    return 1;
  }
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    fprintf(stderr, "failed to rewind %s\n", path);
    return 1;
  }

  bytes = (uint8_t *)malloc((size_t)file_size);
  if (bytes == NULL) {
    fclose(fp);
    fprintf(stderr, "failed to allocate %ld bytes\n", file_size);
    return 1;
  }

  read_len = fread(bytes, 1, (size_t)file_size, fp);
  fclose(fp);
  if (read_len != (size_t)file_size) {
    free(bytes);
    fprintf(stderr, "failed to read %s\n", path);
    return 1;
  }

  *bytes_out = bytes;
  *len_out = read_len;
  return 0;
}

int main(int argc, char **argv) {
  uint8_t *model_bytes = NULL;
  size_t model_len = 0;
  CVI_MODEL_HANDLE model = NULL;
  CVI_TENSOR *inputs = NULL;
  CVI_TENSOR *outputs = NULL;
  int32_t input_num = 0;
  int32_t output_num = 0;
  CVI_RC rc;

  if (argc != 2) {
    fprintf(stderr, "usage: %s /path/to/yolo.cvimodel\n", argv[0]);
    return 1;
  }

  if (read_file(argv[1], &model_bytes, &model_len) != 0) {
    return 1;
  }

  rc = CVI_NN_RegisterModelFromBuffer((const int8_t *)model_bytes, (uint32_t)model_len, &model);
  if (rc != CVI_RC_SUCCESS || model == NULL) {
    fprintf(stderr, "CVI_NN_RegisterModelFromBuffer failed rc=%d\n", rc);
    free(model_bytes);
    return 1;
  }

  rc = CVI_NN_GetInputOutputTensors(model, &inputs, &input_num, &outputs, &output_num);
  if (rc != CVI_RC_SUCCESS) {
    fprintf(stderr, "CVI_NN_GetInputOutputTensors failed rc=%d\n", rc);
    CVI_NN_CleanupModel(model);
    free(model_bytes);
    return 1;
  }

  puts("# Copy the values below into assets/models/yolo.toml");
  puts("");
  puts("[input]");
  if (input_num <= 0) {
    puts("# no inputs reported");
  } else {
    printf("name = \"%s\"\n", inputs[0].name);
    printf("# fmt = %d\n", inputs[0].fmt);
    printf("# qscale = %.9g\n", inputs[0].qscale);
    printf("# zero_point = %d\n", inputs[0].zero_point);
    fputs("# dimensions = ", stdout);
    print_dimensions(&inputs[0].shape);
    putchar('\n');
  }

  for (int32_t i = 0; i < output_num; ++i) {
    puts("");
    puts("[[outputs]]");
    printf("name = \"%s\"\n", outputs[i].name);
    fputs("dimensions = ", stdout);
    print_dimensions(&outputs[i].shape);
    putchar('\n');
    printf("# fmt = %d\n", outputs[i].fmt);
    printf("qscale = %.9g\n", outputs[i].qscale);
    printf("zero_point = %d\n", outputs[i].zero_point);
    puts("# fill stride and anchors manually after matching this output to the YOLO head");
  }

  CVI_NN_CleanupModel(model);
  free(model_bytes);
  return 0;
}
