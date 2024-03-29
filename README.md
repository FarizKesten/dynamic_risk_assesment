# Model that asses risk of clients exiting contracts
Trained using simple logistic regression


# Using App.py

## 1. run the app using script

```bash
    python apicalls.py
```

## 2. run app.py in the terminal

```bash
python app.py
```

## call the api using curl in a new terminal, Examples:

### prediction
```bash
curl -X POST 127.0.0.1:8000/prediction -d '{"file" : "/home/bobby/codes/dynamic_risk_asessment_system/testdata/testdata.csv"}'

## output:
[0, 1, 1, 1, 1]
```

### scoring:
```bash

curl 127.0.0.1:8000/scoring

## output
{
  "score": 0.5714285714285715
}
```

### summarystats:
```bash
curl 127.0.0.1:8000/summarystats
## output
{
  "exited": {
    "mean": 0.5769230769230769,
    "median": 1.0,
    "std": 0.5038314736557788
  },
  "lastmonth_activity": {
    "mean": 165.65384615384616,
    "median": 73.0,
    "std": 284.0332293669447
  },
  "lastyear_activity": {
    "mean": 1502.923076923077,
    "median": 955.0,
    "std": 2192.6449584568304
  },
  "number_of_employees": {
    "mean": 26.884615384615383,
    "median": 14.0,
    "std": 31.353885785435814
  }
}
```

### diagnostics
```bash

curl 127.0.0.1:8000/diagnostics
## output
{
  "dependencies": [
    {
      "current_version": "2.1.1",
      "latest_version": "3.0.0",
      "package": "charset-normalizer"
    },
    {
      "current_version": "8.0.4",
      "latest_version": "8.1.3",
      "package": "click"
    },
    {
      "current_version": "0.10.0",
      "latest_version": "0.11.0",
      "package": "cycler"
    },
    {
      "current_version": "0.63.0",
      "latest_version": "0.86.0",
      "package": "fastapi"
    },
    {
      "current_version": "1.1.2",
      "latest_version": "2.2.2",
      "package": "Flask"
    },
    {
      "current_version": "20.0.4",
      "latest_version": "20.1.0",
      "package": "gunicorn"
    },
    {
      "current_version": "4.11.3",
      "latest_version": "5.0.0",
      "package": "importlib-metadata"
    },
    {
      "current_version": "1.1.0",
      "latest_version": "2.1.2",
      "package": "itsdangerous"
    },
    {
      "current_version": "2.11.3",
      "latest_version": "3.1.2",
      "package": "Jinja2"
    },
    {
      "current_version": "1.0.1",
      "latest_version": "1.2.0",
      "package": "joblib"
    },
    {
      "current_version": "1.3.1",
      "latest_version": "1.4.4",
      "package": "kiwisolver"
    },
    {
      "current_version": "1.1.1",
      "latest_version": "2.1.1",
      "package": "MarkupSafe"
    },
    {
      "current_version": "3.3.4",
      "latest_version": "3.6.2",
      "package": "matplotlib"
    },
    {
      "current_version": "1.20.1",
      "latest_version": "1.23.4",
      "package": "numpy"
    },
    {
      "current_version": "1.2.2",
      "latest_version": "1.5.1",
      "package": "pandas"
    },
    {
      "current_version": "8.1.0",
      "latest_version": "9.3.0",
      "package": "Pillow"
    },
    {
      "current_version": "22.2.2",
      "latest_version": "22.3.1",
      "package": "pip"
    },
    {
      "current_version": "2.4.7",
      "latest_version": "3.0.9",
      "package": "pyparsing"
    },
    {
      "current_version": "2.8.1",
      "latest_version": "2.8.2",
      "package": "python-dateutil"
    },
    {
      "current_version": "2021.1",
      "latest_version": "2022.6",
      "package": "pytz"
    },
    {
      "current_version": "0.24.1",
      "latest_version": "1.1.3",
      "package": "scikit-learn"
    },
    {
      "current_version": "1.6.1",
      "latest_version": "1.9.3",
      "package": "scipy"
    },
    {
      "current_version": "0.11.1",
      "latest_version": "0.12.1",
      "package": "seaborn"
    },
    {
      "current_version": "63.4.1",
      "latest_version": "65.5.1",
      "package": "setuptools"
    },
    {
      "current_version": "1.15.0",
      "latest_version": "1.16.0",
      "package": "six"
    },
    {
      "current_version": "0.0",
      "latest_version": "0.0.post1",
      "package": "sklearn"
    },
    {
      "current_version": "0.13.6",
      "latest_version": "0.21.0",
      "package": "starlette"
    },
    {
      "current_version": "2.1.0",
      "latest_version": "3.1.0",
      "package": "threadpoolctl"
    },
    {
      "current_version": "1.0.1",
      "latest_version": "2.2.2",
      "package": "Werkzeug"
    },
    {
      "current_version": "0.37.1",
      "latest_version": "0.38.4",
      "package": "wheel"
    },
    {
      "current_version": "3.8.0",
      "latest_version": "3.10.0",
      "package": "zipp"
    }
  ],
  "execution_time": [
    1.906566260001273,
    1.0596662600000855
  ],
  "missing_list": {
    "corporation": 0.0,
    "exited": 0.0,
    "lastmonth_activity": 0.0,
    "lastyear_activity": 0.0,
    "number_of_employees": 0.0
  }
}
```



