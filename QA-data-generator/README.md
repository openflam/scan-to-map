# Question-Answer data generator for a given space

To generate the component data for all frames in a dataset:

```
python main.py <dataset_name> --pixel-threshold <num_min_pixles>
```

To call LLM for a single frame and generate data for it:
```
python -m src.call_llm <dataset_name> <frame_name> --model <model_name> --num-questions <num_questions>
```

## Examples:

```
python main.py ProjectLabStudio_NoNeg --pixel-threshold 5000
python -m src.call_llm ProjectLabStudio_NoNeg frame_00001 --model gpt-5.2 --num-questions 10
```