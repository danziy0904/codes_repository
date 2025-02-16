sample_rate = 44100
window_size = 2048
overlap = 672
seq_len = 320
mel_bins = 64

labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
          'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}
