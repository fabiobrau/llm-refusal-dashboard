## Refusal Suppression Dashboard

This is the init instruction file to build a refusal dashboard. The aim is to show a realistic refusal suppresion on an opensource model by steering activations.

The dashboard allows for selecting a model from hugging face, having an entry of the HF_TOKEN, selecting what layers can be steered and selecting the steering amplitude. The steering is made by applying a projection on the inner states that is explained in the refusal.md instruction file.

The dashboard work throught the following step.
1. Fase 1 (Model Choiche) In this phase the user select the model and the activations on harmful and harmless datasets are pre-computed. They will be used to deduce a refusal direction. The model runs on the device that is selectable through a menu (mpi for mac, nvidia device on linux if present). The dataset url are stored in harmprompts.md file

2. Fase 2 (User Experience) Here is the real user experience. The user can select what layers the steering can be applied, and can select the magnitude. The user see a chat box, similar to ollama or gpt (API of external package are allowed to reduce the size of this repo.) The user can also select from a list, a few common harmful request taken from the dataset.


 