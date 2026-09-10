# Data Policy

This repository does not include the IDRT application dataset or generated report
artifacts that embed source imagery.

To run the satellite comparison workflow, place the supplied image files in a
local folder and set:

```powershell
$env:IDRT_DATASET_DIR="path\to\IDRT application dataset"
```

The report generator writes outputs to `reports/` by default. That folder is
ignored because it may contain source or derived imagery. Share generated reports
and figures separately only when the dataset terms allow redistribution.

If the dataset folder is not present, `Open IDRT Pair` shows a dataset policy
message instead of failing. Reviewers can still use `Open Image`, `Open Video`,
or their own local paired imagery to exercise GAIA's general image-processing
tools.
