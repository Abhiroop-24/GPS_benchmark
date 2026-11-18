# GPS Benchmark

Independent implementation of the PathSnap distributed map-matching challenge.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ln -sfn "/home/abhiroop/Documents/ML_hackathon/Student folder" data
```

The supplied dataset is intentionally not committed to this repository.

## Dataset layout

Each trip is stored under `train/<trip_id>` or `student_test/<trip_id>`:

- `<trip_id>.track`: longitude, latitude, timestamp
- `<trip_id>.nodes`: longitude, latitude
- `<trip_id>.arcs`: from-node ID, to-node ID
- `<trip_id>.route`: reference arc IDs

## First milestone

`src/candidates.py`:

- projects each trip into local UTM coordinates;
- creates candidate road segments around each GPS point;
- calculates exact point-to-segment distance;
- aligns GPS observations to the monotonic reference route for supervised labels;
- reports candidate miss and Top-K recall.
