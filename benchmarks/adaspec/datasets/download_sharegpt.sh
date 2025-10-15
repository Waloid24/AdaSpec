# make sure you have huggingface-cli installed
huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered --repo-type dataset --include "ShareGPT_V3_unfiltered_cleaned_split.json"

mv ./ShareGPT_V3_unfiltered_cleaned_split.json ./ShareGPT.json