# mgr_ml_server
Server to run classification ml model for an app.

## To run
```
python -m speciesnet.scripts.run_server --port=8008 --extra_fields=country
```

Now send POST request to it with

```
data = {
    "instances": [
        {
            "filepath": "https://link.to/image.jpg",
            "country": "POL"
        }
    ]
}
```
