event_config = {}
event_config['num_bins'] = 5
event_config['n_frame_steps'] = 40
event_config['CT'] = {
    'CT_mean': 0.2,
    'CT_std': 0.03,
    'CT_min': 0.01
}
event_config['gauss_noise'] = {
    'gauss_mean': 0.0,
    'gauss_std': 0.1
}
event_config['hot_pixel'] = {
    'hot_mean': 0.0,
    'hot_std': 0.2,
    'hot_p': 0.001
}
