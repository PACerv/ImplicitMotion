# Evaluation Models
The evaluation metrics for motion generation require models for feature extraction.

[ACTOR](https://github.com/Mathux/ACTOR) provides a convenient download script to download the weights for the feature extractors used in this work for **HumanAct12, NTU13 and UESTC**. [[link]](https://github.com/Mathux/ACTOR/tree/master/prepare)

Note, that the action recognition models for **HumanAct12** and **NTU13** are originally provided by [Action2Motion](https://github.com/EricGuo5513/action-to-motion/tree/master/model_file)

After downloading, place the models in `ImplicitMotion/model_file` or specify `--path_classifier` when testing.