# mcoc-arena-bot

Project for automating MCOC arena using computer vision and automation tools.

## Project Structure

```
mcoc-arena-bot/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── game_config.yaml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_bot.py
│   │   └── exceptions.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── model.py          # CNN model definition
│   │   ├── trainer.py        # Training logic
│   │   ├── inference.py      # Prediction engine
│   │   └── preprocessing.py  # Image preprocessing
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── screen_recorder.py
│   │   ├── click_logger.py
│   │   └── dataset_manager.py
│   ├── game_controller/
│   │   ├── __init__.py
│   │   ├── navigation.py     # Menu navigation
│   │   ├── arena_manager.py  # Arena logic
│   │   └── actions.py        # PyAutoGUI wrapper
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py
│       └── logging_config.py
├── models/           # Saved CNN models
├── data/
│   ├── raw/         # Raw screenshots
│   ├── processed/   # Preprocessed images
│   └── annotations/ # Labels/click data
├── scripts/
│   ├── train_model.py
│   ├── collect_data.py
│   └── run_bot.py
└── tests/
    ├── __init__.py
    ├── test_vision.py
    ├── test_controller.py
    └── test_integration.py
```

## Usage

Instructions coming soon.