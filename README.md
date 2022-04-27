# Deep Regularity Structure

## Phi41 Model

- Generate Data:
```bash
python parabolic_data.py -N 1000 -k 0.0
python parabolic_data.py -N 1000 -k 0.1
python parabolic_data.py -N 10000 -k 0.0
python parabolic_data.py -N 10000 -k 0.1
```
- Model training:
```bash
python parabolic.py -N 1000 -k 0.0
python parabolic.py -N 1000 -k 0.1
python parabolic.py -N 10000 -k 0.0
python parabolic.py -N 10000 -k 0.1
```