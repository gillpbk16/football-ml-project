# Football Match Prediction System 

ML system for predicting football match outcomes using team statistics and betting market data.

**Performance:** Best model: 54.2% accuracy (vs 33.3% random baseline)

## Tech Stack

* **ML/Data:** scikit-learn, pandas, numpy
* **API:** FastAPI, uvicorn, pydantic  
* **Frontend:** HTML/CSS/JavaScript
* **Testing:** pytest (19 tests)

## Data
[Football-Data.co.uk](https://www.football-data.co.uk/)

## Setup
```bash
git clone https://github.com/gillpbk16/football-ml
cd football-ml
pip install -r requirements.txt

# Train models
python -m src.train_models

# Start API server
cd api && python app.py

# Frontend: frontend/index.html
# API docs: http://localhost:8000/docs
```
## Features
* Rolling team statistics with temporal validation
* Betting odds integration
* Model versioning and comparison
* Production API with A/B testing
* Web interface for predictions

## Testing:

```bash
pytest 
```

## Future Improvements:
* Real-time data integration from sports APIs
* Advanced features: player injuries, weather conditions
* Deep learning models and ensemble methods
