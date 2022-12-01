### Install conda env
``` 
conda env create -f env.yml
```

### Create tmp folder for logging temp files on brain
```
mkdir tmp
```

### After activating conda environment, run:
```
sbatch launch/run.sh
```