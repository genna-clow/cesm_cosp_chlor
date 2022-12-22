# Pre-industrial control simulation 

## Case setup
- Code directory: https://github.com/genna-clow/CESM/tree/cesm2.2.0_satchl
- Case directory: /glade/u/home/gclow/cases/b.e22.B1850.f09_g17.cosp_chlor_30yr
- Grid: f09_g17
- Compset: B1850

## Data storage 
- Raw model outputs: `/glade/scratch/gclow/archive/b.e22.B1850.f09_g17.cosp_chlor_30yr`
- Daily model outputs: `/glade/scratch/gclow/archive/b.e22.B1850.f09_g17.cosp_chlor_30yr/ocn/daily`
- Climatology data: `data/*mean.nc`
- Global means table: `data/global_mean.csv`
- Biome means table: `data/biome_means.csv`
- Dirunal cycle data: `data/monthly_diurnal_cycle.nc`

## File structure
- **preprocessing**: notebooks for converting hourly model output into more manageable data files
    - `calculate_climatologies.ipynb`: save climatology outputs, make climatology maps, calculate global mean and means over biomes
    - `hourly_to_daily_output.ipynb`: take the daily mean mean of the model outputs and save files here: `/glade/scratch/gclow/archive/b.e22.B1850.f09_g17.cosp_chlor_30yr/ocn/daily`
- **analysis**: notebooks for creating final plots and tables
- **data**: output files generated from the `preprocessing` notebooks