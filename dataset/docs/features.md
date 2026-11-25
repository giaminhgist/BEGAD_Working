# BEGAD eye-tracking feature dataset (69 features)

## Full feature table

The table lists the 69 features in dictionary order.

| # | Feature | Category | Definition and formula | Unit | Applicable tasks |
|---:|---|---|---|---|---|
| 1 | `task_duration_s` | session_event | Duration of the task recording represented in the file, in seconds. Formula: D = (max over events of (t_start(e) + GazeEventDuration(e)) - min over events of t_start(e)) / 1000, where t_start(e) is the earliest RecordingTimestamp of the event and events are unique (FixationIndex or SaccadeIndex) records. | s | 1,3,4,5,7,9 |
| 2 | `fixation_count` | session_event | Number of distinct FixationIndex values among Fixation rows. Formula: Number of distinct `FixationIndex` values | count | 1,3,4,5,7,9 |
| 3 | `saccade_count` | session_event | Number of distinct SaccadeIndex values among Saccade rows. Formula: Number of distinct `SaccadeIndex` values | count | 1,3,4,5,7,9 |
| 4 | `fixation_onset_latency_ms` | session_event | Delay from the first retained event to the first fixation onset. Formula: t_start(F_1) - min over all rows of RecordingTimestamp, in ms, where F_1 is the earliest fixation event. | ms | 1,3,4,5,7,9 |
| 5 | `fixation_duration_cv` | fixation_temporal | std(GazeEventDuration, ddof=1) / mean(GazeEventDuration) over unique fixation events. Formula: std(dur, ddof=1) / mean(dur) | ratio | 1,3,4,5,7,9 |
| 6 | `fixation_first_duration_ms` | fixation_temporal | Duration of the first fixation event. Formula: GazeEventDuration of F_1, the earliest fixation event by start time. | ms | 1,3,4,5,7,9 |
| 7 | `fixation_last_duration_ms` | fixation_temporal | Duration of the last fixation event. Formula: GazeEventDuration of F_m, the latest fixation event by start time. | ms | 1,3,4,5,7,9 |
| 8 | `saccade_to_fixation_duration_ratio` | saccade_temporal | Ratio of mean saccade duration to mean fixation duration. Formula: mean(GazeEventDuration over saccade events) / mean(GazeEventDuration over fixation events). | ratio | 1,3,4,5,7,9 |
| 9 | `fixation_x_iqr_norm` | fixation_spatial | Interquartile range of normalized horizontal fixation position. Formula: p75 - p25 of x_i / MediaWidth over unique fixation events (linear interpolation). | normalized (width units) | 1,3,4,5,7,9 |
| 10 | `fixation_centroid_distance_from_center_frac_diag` | fixation_spatial | Distance of the fixation centroid from the media center, as a fraction of the media diagonal. Formula: dx = mean(x_i) - MediaWidth/2; dy = mean(y_i) - MediaHeight/2; value = sqrt(dx^2 + dy^2) / sqrt(MediaWidth^2 + MediaHeight^2). | ratio (fraction of diagonal) | 1,3,4,5,7,9 |
| 11 | `fixation_quadrant_entropy_bits` | fixation_spatial | Shannon entropy of fixation counts across four media quadrants. Formula: Quadrants split at x/MediaWidth = 0.5 and y/MediaHeight = 0.5; p_q = count in quadrant q / m; H = -sum_q p_q log2(p_q), 0*log2(0)=0. | bits | 1,3,4,5,7,9 |
| 12 | `fixation_grid_occupancy_ratio` | fixation_spatial | Fraction of the nine 3x3 grid cells that contain at least one fixation. Formula: count(cells with count > 0) / 9, using the 3x3 grid of fixation_grid_entropy_bits. | ratio | 1,3,4,5,7,9 |
| 13 | `fixation_y_entropy_bits` | fixation_spatial | Shannon entropy of the vertical fixation-position histogram. Formula: Ten equal bins of y_i/MediaHeight over [0,1]; p_b = count/m; H = -sum_b p_b log2(p_b). | bits | 1,3,4,5,7,9 |
| 14 | `fixation_xy_correlation` | fixation_spatial | Pearson correlation between normalized fixation x and y positions. Formula: corr(x_i/MediaWidth, y_i/MediaHeight) over unique fixation events; sample correlation. | unitless (r) | 1,3,4,5,7,9 |
| 15 | `pre_fixation_saccade_amplitude_p10_deg` | saccade_amplitude_proxy | 10th percentile of fixation-attached saccade amplitude. Formula: 10th percentile (linear interpolation) of SaccadicAmplitude over unique fixation events. | deg (export units) | 1,3,4,5,7,9 |
| 16 | `pre_fixation_saccade_amplitude_p90_deg` | saccade_amplitude_proxy | 90th percentile of fixation-attached saccade amplitude. Formula: 90th percentile (linear interpolation) of SaccadicAmplitude over unique fixation events. | deg (export units) | 1,3,4,5,7,9 |
| 17 | `pre_fixation_saccade_amplitude_trend_rho` | saccade_amplitude_proxy | Within-task monotonic trend in fixation-attached saccade amplitude. Formula: Spearman rank correlation between fixation ordinal (1..m) and SaccadicAmplitude. | unitless (rho) | 1,3,4,5,7,9 |
| 18 | `fixation_transition_amplitude_mean_px` | fixation_transition | Mean distance between consecutive fixation positions, in pixels. Formula: For i=1..m-1: dx=x_{i+1}-x_i, dy=y_{i+1}-y_i, d=sqrt(dx^2+dy^2); value = mean(d). Coordinates are FixationPointX/Y (MCSpx). | px | 1,3,4,5,7,9 |
| 19 | `fixation_transition_amplitude_median_px` | fixation_transition | Median distance between consecutive fixation positions. Formula: median of d_i over i=1..m-1, with d_i as in fixation_transition_amplitude_mean_px. | px | 1,3,4,5,7,9 |
| 20 | `fixation_transition_amplitude_p90_px` | fixation_transition | 90th percentile of consecutive-fixation transition distances. Formula: 90th percentile (linear interpolation) of d_i over i=1..m-1. | px | 1,3,4,5,7,9 |
| 21 | `fixation_transition_amplitude_p10_px` | fixation_transition | 10th percentile of consecutive-fixation transition distances. Formula: 10th percentile (linear interpolation) of d_i over i=1..m-1. | px | 1,3,4,5,7,9 |
| 22 | `fixation_transition_dx_sd_px` | fixation_transition | SD of signed horizontal displacements between consecutive fixations. Formula: sample SD (n-1) of dx_i over i=1..m-1; dx_i = x_{i+1}-x_i in MCSpx. | px | 1,3,4,5,7,9 |
| 23 | `fixation_transition_amplitude_trend_rho` | fixation_transition | Within-task monotonic trend in consecutive-fixation transition distance. Formula: Spearman rank correlation between transition ordinal (1..m-1) and d_i. | unitless (rho) | 1,3,4,5,7,9 |
| 24 | `fixation_transition_dy_mean_px` | fixation_transition | Mean signed vertical displacement between consecutive fixations. Formula: mean(dy_i) over i=1..m-1; dy_i = y_{i+1}-y_i in MCSpx. | px | 1,3,4,5,7,9 |
| 25 | `fixation_transition_speed_mean_px_s` | fixation_transition | Mean inter-fixation displacement speed. Formula: mean( d_i / (g_i/1000) ) over transitions with g_i > 0; d_i as above. | px/s | 1,3,4,5,7,9 |
| 26 | `fixation_transition_direction_circular_sd_deg` | fixation_transition | Circular SD of consecutive-fixation transition directions. Formula: R = sqrt(mean(cos theta)^2 + mean(sin theta)^2); SD = sqrt(-2 ln R) radians converted to degrees. | deg | 1,3,4,5,7,9 |
| 27 | `fixation_transition_rightward_proportion` | fixation_transition | Proportion of consecutive-fixation transitions pointing rightward. Formula: count(cos(theta_i) > 0.7071) / number of transitions. | ratio | 1,3,4,5,7,9 |
| 28 | `fixation_revisit_rate` | scanpath_sequence | Proportion of fixations that revisit a previously visited 3x3 grid cell. Formula: fixation_revisit_count / (m - 1). | ratio | 1,3,4,5,7,9 |
| 29 | `fixation_grid_transition_conditional_entropy_bits` | scanpath_sequence | Conditional entropy of the next 3x3 grid cell given the current cell. Formula: For consecutive fixation cell pairs (c_i, c_{i+1}), i=1..m-1: H(next\|current) = -sum_c p(c) sum_{c'} p(c'\|c) log2 p(c'\|c), with p(c) over the m-1 current cells. | bits | 1,3,4,5,7,9 |
| 30 | `scanpath_path_area_norm` | scanpath_sequence | Absolute area enclosed by the polyline through fixation points, as a fraction of media area. Formula: Normalize x by MediaWidth and y by MediaHeight; close the path F_1..F_m; compute abs(0.5 * sum_i (x_i*y_{i+1} - x_{i+1}*y_i)) with (m+1 -> 1); this is already a fraction of media area. | ratio (fraction of media area) | 1,3,4,5,7,9 |
| 31 | `pupil_skewness` | pupil | Fisher-Pearson skewness of two-eye pupil diameter. Formula: g1 = m3/m2^1.5 with mr = (1/n) sum((p - mean)^r), r = 2,3; population moments. | unitless | 1,3,4,5,7,9 |
| 32 | `pupil_fixation_minus_saccade_mm` | pupil | Difference between fixation-period and saccade-period pupil diameter. Formula: pupil_fixation_mean_mm - pupil_saccade_mean_mm. | mm | 1,3,4,5,7,9 |
| 33 | `pupil_late_minus_early_mm` | pupil | Change in two-eye pupil diameter from the early to the late part of the task. Formula: Early = rows with RecordingTimestamp < T0 + 0.25*(Tend-T0); late = rows with RecordingTimestamp >= T0 + 0.75*(Tend-T0); value = mean(p_late) - mean(p_early). | mm | 1,3,4,5,7,9 |
| 34 | `reading_line_change_downward_proportion` | reading_specific | Proportion of line-change-proxy transitions that move downward. Formula: count(dy_i > 0.05*MediaHeight) / reading_line_change_count. | ratio | 7,9 |
| 35 | `kd_downward_line_transition_count` | king_devick_specific | Number of large downward transitions (next-line proxy) on a King-Devick card. Formula: count(dy_i > 0.10*MediaHeight) over i=1..m-1. | count | 1,3,4,5 |
| 36 | `kd_upward_line_transition_rate_per_s` | king_devick_specific | Upward return-transition proxy rate. Formula: kd_upward_line_transition_count / D. | returns/s | 1,3,4,5 |
| 37 | `kd_time_per_line_s` | king_devick_specific | Task time divided by the number of lines entered (proxy). Formula: D / (1 + kd_downward_line_transition_count). | s/line | 1,3,4,5 |
| 38 | `kd_forward_transition_proportion` | king_devick_specific | Proportion of transitions moving rightward on a King-Devick card. Formula: count(dx_i > 0.01*MediaWidth) / (m-1). | ratio | 1,3,4,5 |
| 39 | `kd_downward_line_vertical_displacement_mean_norm` | king_devick_specific | Mean downward line-jump size as a fraction of card height. Formula: mean(dy_i) / MediaHeight over transitions with dy_i > 0.10*MediaHeight. | ratio (fraction of height) | 1,3,4,5 |
| 40 | `kd_upward_line_vertical_displacement_mean_norm` | king_devick_specific | Mean upward backtrack size as a fraction of card height. Formula: mean(\|dy_i\|) / MediaHeight over transitions with dy_i < -0.10*MediaHeight. | ratio (fraction of height) | 1,3,4,5 |
| 41 | `fixation_duration_min` | fixation | min(GazeEventDuration) over unique fixation events. Formula: min(dur) | ms | 1,3,4,5,7,9 |
| 42 | `fixation_time_proportion` | fixation | fixation_time_total / span_ms. Formula: fixation_time_total / span_ms | ratio | 1,3,4,5,7,9 |
| 43 | `fixation_y_span` | fixation | max(FixationPointY) - min(FixationPointY) in MCS px. Formula: max(Y) − min(Y) | px | 1,3,4,5,7,9 |
| 44 | `fixation_count_top_half` | fixation | Number of fixation events with Y < H/2. Formula: Number of events with Y < H/2 | count | 1,3,4,5,7,9 |
| 45 | `fixation_count_left_half` | fixation | Number of fixation events with X < W/2. Formula: Number of events with X < W/2 | count | 1,3,4,5,7,9 |
| 46 | `fixation_count_right_half` | fixation | Number of fixation events with X >= W/2. Formula: Number of events with X ≥ W/2 | count | 1,3,4,5,7,9 |
| 47 | `std_fixation_gap` | fixation | std(start(i+1) - start(i), ddof=1) over consecutive fixation events. Formula: std of those intervals (ddof=1) | ms | 1,3,4,5,7,9 |
| 48 | `fixation_nn_mean_distance` | fixation | Mean over fixation points of the Euclidean distance to the nearest OTHER fixation point. Formula: Mean over fixation points of the Euclidean distance to the nearest *other* fixation point | px | 1,3,4,5,7,9 |
| 49 | `saccade_amplitude_mean` | saccade | mean(SaccadicAmplitude) over fixation events (amplitude is populated on fixation rows and describes the incoming saccade). Formula: mean(amp) | deg | 1,3,4,5,7,9 |
| 50 | `saccade_amplitude_median` | saccade | median(SaccadicAmplitude) over fixation events. Formula: median(amp) | deg | 1,3,4,5,7,9 |
| 51 | `small_saccade_proportion` | saccade | Fraction of fixation events with amplitude < 1 deg. Formula: Fraction of fixation events with amp < 1° | ratio | 1,3,4,5,7,9 |
| 52 | `large_saccade_proportion` | saccade | Fraction of fixation events with amplitude > 10 deg. Formula: Fraction of fixation events with amp > 10° | ratio | 1,3,4,5,7,9 |
| 53 | `saccade_amplitude_total_per_min` | saccade | saccade_amplitude_total / span_min. Formula: saccade_amplitude_total / span_min | deg/min | 1,3,4,5,7,9 |
| 54 | `fixation_to_saccade_time_ratio` | saccade | fixation_time_total / saccade_time_total. Formula: fixation_time_total / saccade_time_total | ratio | 1,3,4,5,7,9 |
| 55 | `gaze_path_length` | gaze_scanpath | Sum of Euclidean distances between consecutive rows' ADCS gaze points. Formula: Σ Euclidean distance between consecutive rows' gaze points | px | 1,3,4,5,7,9 |
| 56 | `gaze_point_std_x` | gaze_scanpath | std(GazePointX, ddof=1) in ADCS px. Formula: std(GazePointX, ddof=1) | px | 1,3,4,5,7,9 |
| 57 | `gaze_cog_shift_x` | gaze_scanpath | mean(GazePointX) of the second half of rows by timestamp minus the first half (signed). Formula: mean(GazePointX) of the second half of the rows (by timestamp) − mean of the first half (signed) | px | 1,3,4,5,7,9 |
| 58 | `scanpath_length_per_min` | gaze_scanpath | fixation_scanpath_length / span_min. Formula: fixation_scanpath_length / span_min | px/min | 1,3,4,5,7,9 |
| 59 | `gaze_centroid_y` | gaze_scanpath | mean(GazePointY) in ADCS px. Formula: mean(GazePointY) | px | 1,3,4,5,7,9 |
| 60 | `gaze_quadrant_prop_bottom_left` | gaze_scanpath | Fraction of rows with X < midX and Y >= midY of the global ADCS box. Formula: Fraction of rows with X < midX and Y ≥ midY | ratio | 1,3,4,5,7,9 |
| 61 | `kd_quadrant_transition_count` | king_devick | Number of consecutive fixation pairs (by start timestamp) whose 2x2 quadrant index changes. Formula: Number of consecutive fixation pairs (by start timestamp) whose quadrant index changes | count | 1,3,4,5 |
| 62 | `fixation_point_mean_y` | fixation | mean(FixationPointY) in MCS px. Formula: mean(Y) | px | 1,3,4,5,7,9 |
| 63 | `fixation_aspect_ratio` | fixation | fixation_x_span / fixation_y_span (NaN if y_span = 0). Formula: fixation_x_span / fixation_y_span (NaN if y_span = 0) | ratio | 1,3,4,5,7,9 |
| 64 | `saccade_time_proportion` | saccade | saccade_time_total / span_ms. Formula: saccade_time_total / span_ms | ratio | 1,3,4,5,7,9 |
| 65 | `reading_return_sweep_amplitude_mean` | reading | Mean incoming amplitude of fixations whose 40-px band is smaller than the previous fixation's band (line returns). Formula: Mean incoming amp of fixations whose band is smaller than the previous fixation's band (line returns) | deg | 7,9 |
| 66 | `fixation_x_progression_slope` | reading | OLS slope of FixationPointX against fixation event order (0..n-1). Formula: OLS slope of FixationPointX against fixation event order | px/event | 7,9 |
| 67 | `fixation_prop_top_third` | reading | Fraction of fixations with Y < H/3. Formula: Fraction of fixations with Y < H/3 (H = 417 for task 7, 917 for task 9) | ratio | 7,9 |
| 68 | `fixation_prop_bottom_third` | reading | Fraction of fixations with Y >= 2H/3. Formula: Fraction of fixations with Y ≥ 2H/3 | ratio | 7,9 |
| 69 | `kd_quadrant_duration_std` | king_devick | For each quadrant with at least one fixation, mean GazeEventDuration; std (ddof=1) of these quadrant means. Formula: Per quadrant with ≥ 1 fixation: mean `GazeEventDuration`; std of these quadrant means (ddof=1) | ms | 1,3,4,5 |
