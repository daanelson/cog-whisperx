#!/bin/bash

ct2-transformers-converter --model $1 --output_dir converted_whisper --copy_files tokenizer.json --quantization float16