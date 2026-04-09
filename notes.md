Todo:
- are there any tools to estimate walltime given an amount of CPU/GPU resources to run code?

- sample slurm batch script

#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=2 --ntasks-per-node=40
#SBATCH --job-name=hello
#SBATCH --account=PZSXXXX

cd $SLURM_SUBMIT_DIR

#setup software environment
module load minicoda..xx

# move input files to compute node
cp atmo-retrieval.py $TMPDIR
python -m atmo_retrieval --planet KELT-20b --mode transmission --epoch 20250601

# copy results back to working directory
cp data_product.csv $SLURM_SUBMIT_DIR

- to submit the slurm batch script
	sbatch <jobscript>
	scancel <jobid>
	scontrol hold <jobid>
	scontrol release <jobid>
	squeue -u <user>
	
	
OSC:
	- modules
		- modules list: see modules currently loaded
		- modules spider/avail: modules available
	- specifying resources in a job script
		- nodes
			- cores per node
			- GPUs
		- walltime: overestimate
		- shorter job may start sooner due to backfill
	- client portal
	- you can switch shells around if needed, so bash, zsh
	- how to submit jobs
		- interactive job
			- sinteractive
			- salloc (more complicated)
		- non-interactive job
			- slurm job script (.sh file)
	- submit jobs in batch system in login nodes
	- transferring files to the cluster
		- small files: transfer to login node owens.osc.edu
		- large files: transfer to file transer node sftp.osc.edu
		- OnDemand drag and drop file transfer up to 10GB files !!! 
		- GLOBUS: large file transfer system, probably don't need?
	- data storage
		- home, 500gb already afforded to me
		- project/ess, available to project PIs by request, shared by all users on a project, backed up daily, 1-5TB standard request -- will need to contact Ji if more storage is needed
		- compute nodes: all data is purged when job quits, so data needs streamed from the compute node into home storage (as they run?)
		- scratch: 100TB quota, 170GB/s, not backed up, purged every 90 days. 
	
	- GPUs by cluster		
		- Owens
			- 160 GPU nodes, NVIDIA Pascal P100 GPU, 1 per node, Intel Xeon E5-2680, 128GB RAM, 28 cores per node, 1 GPU per node
		- Pitzer
			- 32 GPU (16GB) nodes, NVIDIA Volta V100 GPUs, 40 cores per node, 384GB RAM, 2 GPUs per node, 1TB local disk space
			- 42 GPU (32GB) nodes, NVIDIA Volta V100 GPUs, 48 cores per node, 384GB RAM, 2 GPUs per node, 1TB local disk space
			- 4 Dense GPU nodes, NVIDIA Volta V100 GPUs, 48 cores per node, 768GB RAM, 4 GPUs per node, 4 TB local disk space
		- Ascend	
			- quad gpu mode
				- 24 servers (nodes) each with 
					- with 4 NVIDIA A100s each, each with 80GB VRAM
					- 1 TB ram
					- 2 AMD EPYC 7643 CPUs
						- 48 cores per CPU
			- triple gpu mode
				- 84 servers (nodes) 
					- 3 NVIDIA A100s, each with 40GB VRAM
					- 2 AMD EPYC 7H12 processors
						- 64 cores per CPU
					- 512GB RAM
			- dual gpu mode
				- 214 servers (nodes) each with
					- 2 NVIDIA A100s, each with 40GB VRAM
					- 2 AMD EPYC 7H12 CPUs
						- 64 cores per CPU
					- 512GB RAM
			2 login nodes --- IP address: 192.148.247.[180-181]

- limits
	- 168hr for single node jobs
	- 96hr for mutli-node jobs
	- 384 concurrent jobs per user
	- 3080 processor cores in use
	- 1000 jobs in the batch system, running, or queued
	 

notes for marshall meeting

- mccoury phd in using jax and differential programmning for exoplanets
	- obtain the latest code from tori on her fork of atmo-analysis -- marshall said that the better sysrem branch i	is the latest within her fork.
- can go a science direction or coding direction -- prefer to do both
	- science
		- use the existing pipeline with few modifications on emission datasets for quick paper
		- use the new pipeline on transmission datasets -- more work 
