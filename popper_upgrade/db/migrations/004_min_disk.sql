alter table registry.models add column min_disk_gb integer check (min_disk_gb is null or min_disk_gb > 0);
