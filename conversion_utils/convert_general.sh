export ORG="stabilityai"
export NAME="stablelm-tuned-alpha-7b"
export OUTDIR="ct2fast-$NAME"

huggingface-cli repo create "ct2fast-$NAME" --type model -y
git lfs install
git clone "https://huggingface.co/michaelfeil/$OUTDIR"
ct2-transformers-converter --model "$ORG/$NAME" --output_dir "tmp-$OUTDIR" --force  --copy_files tokenizer.json tokenizer_config.json --quantization int8_float16
mv "tmp-$OUTDIR"/* "$OUTDIR"
rm -rf "tmp-$OUTDIR"
cd "$OUTDIR"
git add *
git commit -m "initial commit $NAME to ctranslate2:v3.13.0"
huggingface-cli lfs-enable-largefiles .
git push