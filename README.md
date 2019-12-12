# DTU 02456 Deep Learning course project
The purpose of the project is to implement [this paper](https://arxiv.org/pdf/1708.02182v1.pdf) and use the network to generate text.

Setup is initially based on [this](https://github.com/pytorch/examples/tree/master/word_language_model) PyTorch language model example.

See more info about the course [here](https://kurser.dtu.dk/course/02456).

# How to run
The project is developed using Python 3.6 and PyTorch 0.4.

Training a language model is done in [LanguageModel.ipynb](LanguageModel.ipynb) and text generation can be done in either [GenerateText.ipynb](GenerateText.ipynb) (for multinomial sampling) or [GenerateTextBeamSearch.ipynb](GenerateTextBeamSearch.ipynb) (for beam search).

Choosing the dataset is done in the notebooks.
Here you can also choose which configuration to use. 
This points to an entry in [config.yml](config.yml) that can easily be edited (or a new configuration can be added).

Models are saved using the name specified in [config.yml](config.yml) every time a new best validation perplexity is reached.
This allows easy loading for text generation, for example.
