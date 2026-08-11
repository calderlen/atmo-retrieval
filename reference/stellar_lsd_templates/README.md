# PySME stellar LSD templates

The stellar-velocity estimator deconvolves each PEPSI exposure against a
frozen, continuum-normalized intrinsic PySME spectrum. Template wavelengths
must be vacuum Angstroms in the stellar rest frame.

Generate the KELT-20 template with:

```bash
conda run -n retrieval python scripts/generate_pysme_stellar_template.py \
  --planet KELT-20b \
  --linelist input/vald/CalderLenhart.021791 \
  --output reference/stellar_lsd_templates/kelt20b_pysme_lte_vacuum.npz
```

The raw-only transmission targets use the same generator and their dedicated
VALD Extract Stellar files:

```bash
conda run -n retrieval python scripts/generate_pysme_stellar_template.py \
  --planet HAT-P-11b --parameter-source Recommended \
  --linelist input/vald/CalderLenhart.021798 \
  --output reference/stellar_lsd_templates/hatp11b_pysme_lte_vacuum.npz

conda run -n retrieval python scripts/generate_pysme_stellar_template.py \
  --planet HAT-P-11b --parameter-source Recommended \
  --linelist input/vald/CalderLenhart.021826 \
  --output reference/stellar_lsd_templates/hatp11b_blue_pysme_lte_vacuum.npz

conda run -n retrieval python scripts/generate_pysme_stellar_template.py \
  --planet TOI-1789b --parameter-source Recommended \
  --linelist input/vald/CalderLenhart.021800 \
  --output reference/stellar_lsd_templates/toi1789b_pysme_lte_vacuum.npz

conda run -n retrieval python scripts/generate_pysme_stellar_template.py \
  --planet 'V1298 Tau b' --parameter-source Recommended \
  --linelist input/vald/CalderLenhart.021801 \
  --output reference/stellar_lsd_templates/v1298taub_pysme_lte_vacuum.npz \
  --teff-k 5050 --logg 4.25 --metallicity 0.10 --vmicro-kms 0.85
```

V1298 Tau's explicit atmosphere values are the inputs recorded in
`request_vald_pysme_linelist.py`; its current `Recommended` block does not
contain a finite metallicity, so the generator deliberately cannot infer that
template from `config.py` alone.

The generic HAT-P-11 template covers its red arm (6200--7500 A); the dedicated
`hatp11b_blue` template covers its blue arm (4700--5500 A). Pass the blue
template explicitly with `--template` for a HAT-P-11 blue-arm LSD shadow fit.
TOI-1789's 6200--7500 A template is appropriate for its red-only epoch.

The generator deliberately fixes radial velocity, rotational broadening,
macroturbulent broadening, and instrumental broadening to zero. Those
large-scale effects belong to the empirical broadening kernel recovered from
the observations. Thermal, pressure, natural, and microturbulent line
broadening remain in the PySME spectrum.

Every `.npz` template has a required `.json` provenance sidecar. The velocity
measurement rejects templates without that sidecar or with nonzero `vrad`,
`vsini`, or `vmacro` metadata.
