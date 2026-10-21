# Facts cited in the exp4 write-up

| catalog | key | value | source |
|---|---|---|---|
| industrial | n_items | 3,105 | partA_catalogue.json |
| industrial | identical | 22 items in 11 groups (0.71 %) | partA_catalogue.json |
| industrial | near_dup | 1,291 pairs covering 900 items (29.0 %) | partA_catalogue.json |
| industrial | spectrum | participation ratio 49; top 10/64/256 PCs 38 % / 70 % / 89 % | partA_catalogue.json |
| industrial | rate_rqkmeans | 6.3–44.2 % | b_tables.collision_rate (9 maps) |
| industrial | rate_rqvae | 0.7–2.5 % | b_tables.collision_rate (9 maps) |
| industrial | rate_MQ | 19.0–55.9 % | b_tables.collision_rate (9 maps) |
| industrial | rate_3x128 | RQ-KMeans 44.2 % / RQ-VAE 0.7 % / MQ 55.9 % | b_tables |
| industrial | var_alone_rqkmeans_4x128 | 53, 16, 15, 12 % | m1_digits.var_alone, digits 1-4 |
| industrial | var_given_rqkmeans_4x128 | 52.8, 25.7, 17.3, 2.8 % | m1_digits.var_given_prefix, digits 1-4 |
| industrial | var_alone_rqvae_4x128 | 29, 20, 12, 9 % | m1_digits.var_alone, digits 1-4 |
| industrial | var_given_rqvae_4x128 | 29.2, 48.0, 21.9, 0.9 % | m1_digits.var_given_prefix, digits 1-4 |
| industrial | var_alone_MQ_4x128 | 38, 38, 38, 38 % | m1_digits.var_alone, digits 1-4 |
| industrial | var_given_MQ_4x128 | 37.7, 31.2, 17.0, 7.7 % | m1_digits.var_given_prefix, digits 1-4 |
| industrial | mq_var_alone_all | 36–57 % | m1_digits.var_alone, MQ, all digits of 9 maps |
| industrial | mq_var_ratio | 1.01–1.11 | max/min single-digit share per MQ map |
| industrial | rk_d1_alone | 53–69 % | m1_digits, RQ-KMeans digit 1, 9 maps |
| industrial | rk_later_alone | 9–26 % | m1_digits, RQ-KMeans digits >= 2 |
| industrial | rv_d1_alone | 25–37 % | m1_digits, RQ-VAE digit 1 |
| industrial | rv_d1_given | 25–37 % | m1_digits, RQ-VAE digit 1 |
| industrial | rv_d2_given | 43–59 % | m1_digits, RQ-VAE digit 2 given digit 1 |
| industrial | rv_d2_beats_d1 | 9/9 | RQ-VAE maps where digit 2 given digit 1 explains more than digit 1 |
| industrial | rv_late_given_max | 1.5 % | m1_digits, RQ-VAE digits 4-5 given prefix, max |
| industrial | rv_util_d1 | 3–45 % (9/9 at <= 50 %) | m1_digits.utilization, RQ-VAE digit 1 |
| industrial | rk_util | 100 % | m1_digits.utilization, RQ-KMeans all digits |
| industrial | mq_util | 85–99 % | m1_digits.utilization, MQ all digits |
| industrial | purity3_rqkmeans | 97.0 % (null 44.8 %; 43 % of items still grouped) | m1_depths.cat_l1_purity, prefix length 3, rqkmeans 4x128 |
| industrial | purity3_rqvae | 98.6 % (null 49.9 %; 23 % of items still grouped) | m1_depths.cat_l1_purity, prefix length 3, rqvae 4x128 |
| industrial | purity3_MQ | 79.2 % (null 37.9 %; 55 % of items still grouped) | m1_depths.cat_l1_purity, prefix length 3, MQ 4x128 |
| industrial | purity3_residual_all | 90.7–99.3 % | m1_depths.cat_l1_purity, residual maps D>=4, length 3 |
| industrial | purity3_mq_all | 79.0–94.6 % | m1_depths.cat_l1_purity, MQ, length 3 |
| industrial | ri_5x128 | RQ-KMeans 0.96 / RQ-VAE 1.00 / MQ 0.82 | m1_depths.refinement_index, 5x128, length 5 |
| industrial | unique3_rqvae_4x128 | 77.1 % | m1_depths.frac_items_in_singletons, RQ-VAE 4x128, length 3 |
| industrial | alone_last_rqkmeans | 55.8–93.7 % | m1_depths.frac_items_in_singletons at full length |
| industrial | alone_last_rqvae | 97.5–99.3 % | m1_depths.frac_items_in_singletons at full length |
| industrial | alone_last_MQ | 44.1–81.0 % | m1_depths.frac_items_in_singletons at full length |
| industrial | bits_budget | 11.6 | log2(items) |
| industrial | bits_full_rqkmeans | 10.8–11.5 | m1_digits.cum_bits at the last digit |
| industrial | bits_full_rqvae | 11.5–11.6 | m1_digits.cum_bits at the last digit |
| industrial | bits_full_MQ | 10.4–11.3 | m1_digits.cum_bits at the last digit |
| industrial | ami_rqkmeans_4x128 | 0.52, 0.14, 0.24, 0.38† | m1_ami.conditional_ami (cat_l1), digits 1-4; † = unsupported |
| industrial | ami_rqvae_4x128 | 0.50, 0.19, 0.25†, n/a† | m1_ami.conditional_ami (cat_l1), digits 1-4; † = unsupported |
| industrial | ami_MQ_4x128 | 0.32, 0.19, 0.14, 0.10 | m1_ami.conditional_ami (cat_l1), digits 1-4; † = unsupported |
| industrial | floor | 0.71 % | d_loop.identical_floor |
| industrial | rv_pre | 5.5–19.8 % | d_loop.rate_start (RQ-VAE pre-reassignment) |
| industrial | rv_pub | 0.71–2.54 % | d_loop.rate_published (RQ-VAE) |
| industrial | rv_changed | 6–20 % | d_loop.share_last_digit_changed (RQ-VAE) |
| industrial | rv_loop_reproduces | 9/9 tables item for item (min 100.00 %) | d_loop.loop_equals_published_items |
| industrial | rv_at_floor | 8/9 | RQ-VAE published within 0.5 pp of floor |
| industrial | rk_loop | 6.3–44.2 % → 0.71–0.84 % | d_loop rate_start → rate_after_loop |
| industrial | rk_loop_changed | 6–44 % | d_loop.share_last_digit_changed |
| industrial | rk_loop_notfloor | none | d_loop: tables left > 0.5 pp above the floor |
| industrial | rk_loop_cost | 1.0–3.0x | moved items' added last-level sq. distance / table mean |
| industrial | mq_loop | 19.0–55.9 % → 0.71–1.35 % | d_loop rate_start → rate_after_loop |
| industrial | mq_loop_changed | 19–56 % | d_loop.share_last_digit_changed |
| industrial | mq_loop_notfloor | MQ_4codebook_128 1.35 % | d_loop: tables left > 0.5 pp above the floor |
| industrial | mq_loop_cost | 1.3–3.3x | moved items' added last-level sq. distance / table mean |
| industrial | nd_deepest | 5×128 71 %, 5×256 79 %, 5×512 94 % | b_tables.nd_share_collided |
| industrial | nd_trend | W=128: 46 → 62 → 71; W=256: 58 → 74 → 79; W=512: 75 → 90 → 94 | b_tables.nd_share_collided, depths 3→4→5 (%) |
| industrial | nd_mq_deep | 49–65 % | b_tables, MQ depth 5 |
| industrial | identical_split | 27/27 | b_tables.identical_pairs_split == 0 |
| industrial | closest_mate | 70–100 % | b_tables.share_mate_in_closest_0.1pct (RQ-KMeans) |
| industrial | hub_share | 9–21 % | hub.hub_share, levels >= 2 |
| industrial | hub_rank0 | 12/12 | hub rank 0 count |
| industrial | radius_spearman | −0.66 to −0.40 | b_tables.radius_spearman |
| industrial | repeat_rate | 0.032–0.076 (1/K = 0.002–0.008) | b_tables.repeat_rate_mean (RQ-KMeans) |
| industrial | mq_nmi | 0.50–0.70 | b_tables.mean_pairwise_nmi |
| industrial | rk_nmi | 0.28–0.59 | b_tables.mean_pairwise_nmi |
| industrial | rv_nmi | 0.30–0.56 | b_tables.mean_pairwise_nmi |
| industrial | mq_obs_n1 | at least 8x; N1 is 0 in 4 of 9 maps | b_tables.obs_over_n1 |
| industrial | pair_ami_rqkmeans | 0.06–0.11 | m1_pairs: per-map mean pairwise AMI (chance-corrected) |
| industrial | pair_nmi_rqkmeans | 0.28–0.59 | b_tables.mean_pairwise_nmi (plug-in, biased) |
| industrial | pair_ami_rqvae | 0.06–0.13 | m1_pairs: per-map mean pairwise AMI (chance-corrected) |
| industrial | pair_nmi_rqvae | 0.30–0.56 | b_tables.mean_pairwise_nmi (plug-in, biased) |
| industrial | pair_ami_MQ | 0.29–0.33 | m1_pairs: per-map mean pairwise AMI (chance-corrected) |
| industrial | pair_nmi_MQ | 0.50–0.70 | b_tables.mean_pairwise_nmi (plug-in, biased) |
| industrial | mq_3x128_n1 | 55.9 % observed, 6.8 % with independent digits | b_tables MQ 3x128 |
| industrial | mq_init | 9/9 models; parameter std / uniform std 0.88–1.09 | mq_models |
| industrial | c_per-level seed | W=128: +12.9 / +1.5 (ranges overlap), W=256: +4.8 / +0.6 (ranges overlap), W=512: +2.4 (ranges overlap) / -0.7 (ranges overlap) | c_summary paired diff (pp), depth 3 / depth 5, vs native |
| industrial | c_scale-norm | W=128: +5.4 / -2.0 (ranges overlap), W=256: -0.5 (ranges overlap) / -2.9, W=512: -1.4 (ranges overlap) / -1.6 | c_summary paired diff (pp), depth 3 / depth 5, vs native |
| industrial | c_L2 | W=128: -0.5 (ranges overlap) / +0.5 (ranges overlap), W=256: -0.1 (ranges overlap) / -0.2 (ranges overlap), W=512: +0.3 (ranges overlap) / -0.5 (ranges overlap) | c_summary paired diff (pp), depth 3 / depth 5, vs native |
| industrial | c_PCA-256-white | W=128: +13.3 / +3.6, W=256: +3.1 / +0.1 (ranges overlap), W=512: -1.1 (ranges overlap) / -0.7 (ranges overlap) | c_summary paired diff (pp), depth 3 / depth 5, vs PCA-256 |
| industrial | c_PCA-64-white | W=128: -9.4 / -5.1, W=256: -6.2 / -3.0, W=512: -2.7 / -1.0 | c_summary paired diff (pp), depth 3 / depth 5, vs PCA-256 |
| industrial | c_native_vs_pub | 3×128: refit 46.3 % vs published 44.2 %; 5×128: refit 22.1 % vs published 21.2 %; 3×512: refit 16.1 % vs published 16.6 %; 5×512: refit 6.5 % vs published 6.3 % | c_summary native mean over 5 seeds vs b_tables |
| industrial | zcr_n | 21 | zcr rows |
| industrial | zcr_infeasible | none | tables.json zcr state |
| industrial | zcr_below5 | 6 | tables.json zcr == 'below 5%' |
| industrial | zcr_changed_rqkmeans | 3.6–28.9 % | zcr.changed_tail_share |
| industrial | zcr_cost_rqkmeans | 0.7–5.1 % | zcr.sq_dist_rel_increase |
| industrial | zcr_changed_rqvae | 0.4 % | zcr.changed_tail_share |
| industrial | zcr_cost_rqvae | −0.4 to −0.2 % | zcr.sq_dist_rel_increase |
| industrial | zcr_changed_MQ | 12.2–40.4 % | zcr.changed_tail_share |
| industrial | zcr_cost_MQ | 4.8–13.2 % | zcr.sq_dist_rel_increase |
| industrial | zcr_rk3x128 | 29 % of last digits; last-level sq. distance +5.1 % | zcr rqkmeans 3x128 |
| industrial | x_targets | 6,297 | x_exposure |
| industrial | x_repeat | 5.0 % | x_exposure.test_repeat |
| industrial | x_partner_rqkmeans | 4.6–10.8 % | x_exposure.test_sid_partner |
| industrial | x_collided_rqkmeans | 11–49 % | x_exposure.test_targets_collided |
| industrial | x_inter_rqkmeans | 11.1–51.6 % | x_exposure.interactions_on_collided_items |
| industrial | x_partner_rqvae | 3.6–3.9 % | x_exposure.test_sid_partner |
| industrial | x_collided_rqvae | 4–6 % | x_exposure.test_targets_collided |
| industrial | x_inter_rqvae | 4.3–6.0 % | x_exposure.interactions_on_collided_items |
| industrial | x_partner_MQ | 7.7–10.0 % | x_exposure.test_sid_partner |
| industrial | x_collided_MQ | 25–61 % | x_exposure.test_targets_collided |
| industrial | x_inter_MQ | 26.6–61.3 % | x_exposure.interactions_on_collided_items |
| industrial | x_rk3x128 | SID partner 10.8 %, variant 2.1 % (ZCR twin: 0.0 % / 10.6 %) | x_exposure rqkmeans 3x128 |
| industrial | x_new | 82.1–84.4 % | x_exposure.test_new |
| industrial | lat_dead | 71–496 of 128–512 | x_latent level 0 dead_codewords |
| industrial | lat_norm | unused 5.5–5.6 vs used 13.6–29.2 | x_latent level 0 codeword norms (medians) |
| industrial | lat_var | 71–84 % | x_latent level 0 latent variance explained by digit 1 |
| industrial | naming_d1 | RQ-KMeans 0.96–0.99 / RQ-VAE 0.76–0.83 / MQ 0.79–0.88 | m5_naming median held-out AUC, digit 1, 9 maps each |
| industrial | copurchase | RQ-KMeans 23–52x / RQ-VAE 7–17x / MQ 5–16x | m5_copurchase lift at prefix length 1 |
| industrial | old_rk_d1 | 100 % (8 tables; the old 5×512 was a separate balanced fit: 0 %) | r_old_vs_v3 digit1_agree RQ-KMeans |
| industrial | old_rk_d2 | W=128 100 % / W=256 89 % / W=512 78 % | r_old_vs_v3 digit2_agree |
| industrial | old_focal | RQ-KMeans 3×128 same full SID 33 %; RQ-VAE 4×128 first-digit ARI 0.40; MQ first-digit ARI 0.07–0.12 | r_old_vs_v3 |
| office | n_items | 17,696 | partA_catalogue.json |
| office | identical | 117 items in 57 groups (0.66 %) | partA_catalogue.json |
| office | near_dup | 8,874 pairs covering 7,057 items (39.9 %) | partA_catalogue.json |
| office | spectrum | participation ratio 37; top 10/64/256 PCs 42 % / 71 % / 88 % | partA_catalogue.json |
| office | rate_rqkmeans | 12.6–57.7 % | b_tables.collision_rate (9 maps) |
| office | rate_rqvae | 0.7–1.4 % | b_tables.collision_rate (9 maps) |
| office | rate_MQ | 15.6–69.6 % | b_tables.collision_rate (9 maps) |
| office | rate_3x128 | RQ-KMeans 57.7 % / RQ-VAE 1.4 % / MQ 69.6 % | b_tables |
| office | var_alone_rqkmeans_4x128 | 53, 19, 13, 11 % | m1_digits.var_alone, digits 1-4 |
| office | var_given_rqkmeans_4x128 | 52.7, 21.2, 18.2, 5.6 % | m1_digits.var_given_prefix, digits 1-4 |
| office | var_alone_rqvae_4x128 | 42, 7, 3, 3 % | m1_digits.var_alone, digits 1-4 |
| office | var_given_rqvae_4x128 | 41.5, 38.2, 18.3, 1.9 % | m1_digits.var_given_prefix, digits 1-4 |
| office | var_alone_MQ_4x128 | 36, 38, 35, 36 % | m1_digits.var_alone, digits 1-4 |
| office | var_given_MQ_4x128 | 35.6, 23.4, 19.1, 10.6 % | m1_digits.var_given_prefix, digits 1-4 |
| office | mq_var_alone_all | 34–47 % | m1_digits.var_alone, MQ, all digits of 9 maps |
| office | mq_var_ratio | 1.02–1.09 | max/min single-digit share per MQ map |
| office | rk_d1_alone | 53–63 % | m1_digits, RQ-KMeans digit 1, 9 maps |
| office | rk_later_alone | 9–21 % | m1_digits, RQ-KMeans digits >= 2 |
| office | rv_d1_alone | 34–46 % | m1_digits, RQ-VAE digit 1 |
| office | rv_d1_given | 34–46 % | m1_digits, RQ-VAE digit 1 |
| office | rv_d2_given | 38–46 % | m1_digits, RQ-VAE digit 2 given digit 1 |
| office | rv_d2_beats_d1 | 5/9 | RQ-VAE maps where digit 2 given digit 1 explains more than digit 1 |
| office | rv_late_given_max | 2.2 % | m1_digits, RQ-VAE digits 4-5 given prefix, max |
| office | rv_util_d1 | 4–100 % (7/9 at <= 50 %) | m1_digits.utilization, RQ-VAE digit 1 |
| office | rk_util | 100 % | m1_digits.utilization, RQ-KMeans all digits |
| office | mq_util | 98–100 % | m1_digits.utilization, MQ all digits |
| office | purity3_rqkmeans | 95.3 % (null 40.8 %; 56 % of items still grouped) | m1_depths.cat_l2_purity, prefix length 3, rqkmeans 4x128 |
| office | purity3_rqvae | 96.7 % (null 46.4 %; 33 % of items still grouped) | m1_depths.cat_l2_purity, prefix length 3, rqvae 4x128 |
| office | purity3_MQ | 79.8 % (null 35.2 %; 69 % of items still grouped) | m1_depths.cat_l2_purity, prefix length 3, MQ 4x128 |
| office | purity3_residual_all | 95.3–97.2 % | m1_depths.cat_l2_purity, residual maps D>=4, length 3 |
| office | purity3_mq_all | 78.4–90.8 % | m1_depths.cat_l2_purity, MQ, length 3 |
| office | ri_5x128 | RQ-KMeans 0.93 / RQ-VAE 1.00 / MQ 0.75 | m1_depths.refinement_index, 5x128, length 5 |
| office | unique3_rqvae_4x128 | 65.8 % | m1_depths.frac_items_in_singletons, RQ-VAE 4x128, length 3 |
| office | alone_last_rqkmeans | 42.3–87.4 % | m1_depths.frac_items_in_singletons at full length |
| office | alone_last_rqvae | 98.6–99.3 % | m1_depths.frac_items_in_singletons at full length |
| office | alone_last_MQ | 30.4–84.4 % | m1_depths.frac_items_in_singletons at full length |
| office | bits_budget | 14.1 | log2(items) |
| office | bits_full_rqkmeans | 12.9–13.9 | m1_digits.cum_bits at the last digit |
| office | bits_full_rqvae | 14.1 | m1_digits.cum_bits at the last digit |
| office | bits_full_MQ | 12.0–13.9 | m1_digits.cum_bits at the last digit |
| office | ami_rqkmeans_4x128 | 0.56, 0.13, 0.13, 0.07 | m1_ami.conditional_ami (cat_l2), digits 1-4; † = unsupported |
| office | ami_rqvae_4x128 | 0.50, 0.16, 0.15, 0.00† | m1_ami.conditional_ami (cat_l2), digits 1-4; † = unsupported |
| office | ami_MQ_4x128 | 0.36, 0.20, 0.13, 0.15 | m1_ami.conditional_ami (cat_l2), digits 1-4; † = unsupported |
| office | floor | 0.66 % | d_loop.identical_floor |
| office | rv_pre | 7.3–29.9 % | d_loop.rate_start (RQ-VAE pre-reassignment) |
| office | rv_pub | 0.66–1.37 % | d_loop.rate_published (RQ-VAE) |
| office | rv_changed | 7–30 % | d_loop.share_last_digit_changed (RQ-VAE) |
| office | rv_loop_reproduces | 6/9 tables item for item (min 99.99 %) | d_loop.loop_equals_published_items |
| office | rv_at_floor | 8/9 | RQ-VAE published within 0.5 pp of floor |
| office | rk_loop | 12.6–57.7 % → 0.66–3.21 % | d_loop rate_start → rate_after_loop |
| office | rk_loop_changed | 13–58 % | d_loop.share_last_digit_changed |
| office | rk_loop_notfloor | rqkmeans_3codebook_128 3.21 % | d_loop: tables left > 0.5 pp above the floor |
| office | rk_loop_cost | 0.7–1.7x | moved items' added last-level sq. distance / table mean |
| office | mq_loop | 15.6–69.6 % → 0.62–9.67 % | d_loop rate_start → rate_after_loop |
| office | mq_loop_changed | 16–70 % | d_loop.share_last_digit_changed |
| office | mq_loop_notfloor | MQ_3codebook_128 9.67 %, MQ_3codebook_256 1.55 %, MQ_4codebook_128 3.41 %, MQ_5codebook_128 2.14 % | d_loop: tables left > 0.5 pp above the floor |
| office | mq_loop_cost | 0.9–2.8x | moved items' added last-level sq. distance / table mean |
| office | nd_deepest | 5×128 68 %, 5×256 76 %, 5×512 84 % | b_tables.nd_share_collided |
| office | nd_trend | W=128: 39 → 54 → 68; W=256: 49 → 65 → 76; W=512: 59 → 77 → 84 | b_tables.nd_share_collided, depths 3→4→5 (%) |
| office | nd_mq_deep | 33–55 % | b_tables, MQ depth 5 |
| office | identical_split | 27/27 | b_tables.identical_pairs_split == 0 |
| office | closest_mate | 75–100 % | b_tables.share_mate_in_closest_0.1pct (RQ-KMeans) |
| office | hub_share | 3–10 % | hub.hub_share, levels >= 2 |
| office | hub_rank0 | 10/12 | hub rank 0 count |
| office | radius_spearman | −0.60 to −0.16 | b_tables.radius_spearman |
| office | repeat_rate | 0.022–0.046 (1/K = 0.002–0.008) | b_tables.repeat_rate_mean (RQ-KMeans) |
| office | mq_nmi | 0.41–0.53 | b_tables.mean_pairwise_nmi |
| office | rk_nmi | 0.16–0.34 | b_tables.mean_pairwise_nmi |
| office | rv_nmi | 0.12–0.36 | b_tables.mean_pairwise_nmi |
| office | mq_obs_n1 | at least 5x; N1 is 0 in 2 of 9 maps | b_tables.obs_over_n1 |
| office | pair_ami_rqkmeans | 0.08–0.15 | m1_pairs: per-map mean pairwise AMI (chance-corrected) |
| office | pair_nmi_rqkmeans | 0.16–0.34 | b_tables.mean_pairwise_nmi (plug-in, biased) |
| office | pair_ami_rqvae | 0.04–0.09 | m1_pairs: per-map mean pairwise AMI (chance-corrected) |
| office | pair_nmi_rqvae | 0.12–0.36 | b_tables.mean_pairwise_nmi (plug-in, biased) |
| office | pair_ami_MQ | 0.29–0.35 | m1_pairs: per-map mean pairwise AMI (chance-corrected) |
| office | pair_nmi_MQ | 0.41–0.53 | b_tables.mean_pairwise_nmi (plug-in, biased) |
| office | mq_3x128_n1 | 69.6 % observed, 14.2 % with independent digits | b_tables MQ 3x128 |
| office | mq_init | 9/9 models; parameter std / uniform std 0.88–1.09 | mq_models |
| office | c_per-level seed | W=128: +0.9 (ranges overlap) / +1.0 (ranges overlap), W=256: +3.1 (ranges overlap) / -0.0 (ranges overlap), W=512: +3.2 / -1.0 (ranges overlap) | c_summary paired diff (pp), depth 3 / depth 5, vs native |
| office | c_scale-norm | W=128: -3.4 (ranges overlap) / +0.8 (ranges overlap), W=256: -0.2 (ranges overlap) / -1.5, W=512: -0.2 (ranges overlap) / -1.4 | c_summary paired diff (pp), depth 3 / depth 5, vs native |
| office | c_L2 | W=128: +0.4 (ranges overlap) / +0.1 (ranges overlap), W=256: +0.2 (ranges overlap) / +0.2 (ranges overlap), W=512: +0.1 (ranges overlap) / +0.2 (ranges overlap) | c_summary paired diff (pp), depth 3 / depth 5, vs native |
| office | c_PCA-256-white | W=128: +14.9 / +5.0, W=256: +7.4 / +2.6, W=512: +3.4 / +1.7 | c_summary paired diff (pp), depth 3 / depth 5, vs PCA-256 |
| office | c_PCA-64-white | W=128: -14.5 / -5.2, W=256: -10.0 / -3.3, W=512: -6.4 / -2.5 | c_summary paired diff (pp), depth 3 / depth 5, vs PCA-256 |
| office | c_native_vs_pub | 3×128: refit 57.8 % vs published 57.7 %; 5×128: refit 19.9 % vs published 20.2 %; 3×512: refit 29.7 % vs published 30.1 %; 5×512: refit 12.4 % vs published 12.6 % | c_summary native mean over 5 seeds vs b_tables |
| office | zcr_n | 18 | zcr rows |
| office | zcr_infeasible | MQ_3codebook_128 (largest 2-digit prefix group: 530 items), MQ_4codebook_128 (largest 3-digit prefix group: 355 items), MQ_5codebook_128 (largest 4-digit prefix group: 249 items) | tables.json zcr state |
| office | zcr_below5 | 6 | tables.json zcr == 'below 5%' |
| office | zcr_changed_rqkmeans | 7.5–41.0 % | zcr.changed_tail_share |
| office | zcr_cost_rqkmeans | 0.7–5.2 % | zcr.sq_dist_rel_increase |
| office | zcr_changed_rqvae | 0.4–0.5 % | zcr.changed_tail_share |
| office | zcr_cost_rqvae | −0.3 to −0.2 % | zcr.sq_dist_rel_increase |
| office | zcr_changed_MQ | 9.9–41.0 % | zcr.changed_tail_share |
| office | zcr_cost_MQ | 2.2–11.8 % | zcr.sq_dist_rel_increase |
| office | zcr_rk3x128 | 41 % of last digits; last-level sq. distance +5.2 % | zcr rqkmeans 3x128 |
| office | x_targets | 61,564 | x_exposure |
| office | x_repeat | 5.6 % | x_exposure.test_repeat |
| office | x_partner_rqkmeans | 1.9–5.3 % | x_exposure.test_sid_partner |
| office | x_collided_rqkmeans | 14–63 % | x_exposure.test_targets_collided |
| office | x_inter_rqkmeans | 14.8–64.4 % | x_exposure.interactions_on_collided_items |
| office | x_partner_rqvae | 0.6 % | x_exposure.test_sid_partner |
| office | x_collided_rqvae | 1–2 % | x_exposure.test_targets_collided |
| office | x_inter_rqvae | 1.4–2.0 % | x_exposure.interactions_on_collided_items |
| office | x_partner_MQ | 1.9–6.0 % | x_exposure.test_sid_partner |
| office | x_collided_MQ | 18–74 % | x_exposure.test_targets_collided |
| office | x_inter_MQ | 19.7–74.8 % | x_exposure.interactions_on_collided_items |
| office | x_rk3x128 | SID partner 5.3 %, variant 2.5 % (ZCR twin: 0.0 % / 5.6 %) | x_exposure rqkmeans 3x128 |
| office | x_new | 85.7–88.8 % | x_exposure.test_new |
| office | lat_dead | 0–490 of 128–512 | x_latent level 0 dead_codewords |
| office | lat_norm | unused 5.2–5.5 vs used 24.5–36.6 | x_latent level 0 codeword norms (medians) |
| office | lat_var | 68–76 % | x_latent level 0 latent variance explained by digit 1 |
| office | naming_d1 | RQ-KMeans 0.96–0.99 / RQ-VAE 0.85–0.91 / MQ 0.84–0.91 | m5_naming median held-out AUC, digit 1, 9 maps each |
| office | copurchase | RQ-KMeans 30–76x / RQ-VAE 8–26x / MQ 5–15x | m5_copurchase lift at prefix length 1 |
