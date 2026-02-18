# meson_embed/meson.build - バイナリリソース埋め込み

embed_script = files('embed_resource.sh')

resource_specs = [
  ['nnedi3_weights', '../resource/nnedi3_weights.bin', 'resource_nnedi3_weights_bin'],
  ['perf_monitor', '../PerfMonitor/perf_monitor.pyw', 'PerfMonitor_perf_monitor_pyw'],
  ['rgy_filter_colorspace_func', '../QSVPipeline/rgy_filter_colorspace_func.h', 'QSVPipeline_rgy_filter_colorspace_func_h'],
  ['clrng_clh', '../clRNG/src/include/clRNG/clRNG.clh', 'clRNG_src_include_clRNG_clRNG_clh'],
  ['clrng_mrg31k3p_clh', '../clRNG/src/include/clRNG/mrg31k3p.clh', 'clRNG_src_include_clRNG_mrg31k3p_clh'],
  ['clrng_private_mrg31k3p_ch', '../clRNG/src/include/clRNG/private/mrg31k3p.c.h', 'clRNG_src_include_clRNG_private_mrg31k3p_c_h'],
  ['rgy_dummy_load_cl', '../QSVPipeline/rgy_dummy_load.cl', 'QSVPipeline_rgy_dummy_load_cl'],
  ['rgy_filter_cl', '../QSVPipeline/rgy_filter.cl', 'QSVPipeline_rgy_filter_cl'],
  ['rgy_filter_afs_analyze_cl', '../QSVPipeline/rgy_filter_afs_analyze.cl', 'QSVPipeline_rgy_filter_afs_analyze_cl'],
  ['rgy_filter_afs_filter_cl', '../QSVPipeline/rgy_filter_afs_filter.cl', 'QSVPipeline_rgy_filter_afs_filter_cl'],
  ['rgy_filter_afs_merge_cl', '../QSVPipeline/rgy_filter_afs_merge.cl', 'QSVPipeline_rgy_filter_afs_merge_cl'],
  ['rgy_filter_afs_synthesize_cl', '../QSVPipeline/rgy_filter_afs_synthesize.cl', 'QSVPipeline_rgy_filter_afs_synthesize_cl'],
  ['rgy_filter_convolution3d_cl', '../QSVPipeline/rgy_filter_convolution3d.cl', 'QSVPipeline_rgy_filter_convolution3d_cl'],
  ['rgy_filter_curves_cl', '../QSVPipeline/rgy_filter_curves.cl', 'QSVPipeline_rgy_filter_curves_cl'],
  ['rgy_filter_deband_cl', '../QSVPipeline/rgy_filter_deband.cl', 'QSVPipeline_rgy_filter_deband_cl'],
  ['rgy_filter_deband_gen_rand_cl', '../QSVPipeline/rgy_filter_deband_gen_rand.cl', 'QSVPipeline_rgy_filter_deband_gen_rand_cl'],
  ['rgy_filter_decimate_cl', '../QSVPipeline/rgy_filter_decimate.cl', 'QSVPipeline_rgy_filter_decimate_cl'],
  ['rgy_filter_decomb_cl', '../QSVPipeline/rgy_filter_decomb.cl', 'QSVPipeline_rgy_filter_decomb_cl'],
  ['rgy_filter_delogo_cl', '../QSVPipeline/rgy_filter_delogo.cl', 'QSVPipeline_rgy_filter_delogo_cl'],
  ['rgy_filter_denoise_dct_cl', '../QSVPipeline/rgy_filter_denoise_dct.cl', 'QSVPipeline_rgy_filter_denoise_dct_cl'],
  ['rgy_filter_denoise_fft3d_cl', '../QSVPipeline/rgy_filter_denoise_fft3d.cl', 'QSVPipeline_rgy_filter_denoise_fft3d_cl'],
  ['rgy_filter_denoise_knn_cl', '../QSVPipeline/rgy_filter_denoise_knn.cl', 'QSVPipeline_rgy_filter_denoise_knn_cl'],
  ['rgy_filter_denoise_nlmeans_cl', '../QSVPipeline/rgy_filter_denoise_nlmeans.cl', 'QSVPipeline_rgy_filter_denoise_nlmeans_cl'],
  ['rgy_filter_denoise_pmd_cl', '../QSVPipeline/rgy_filter_denoise_pmd.cl', 'QSVPipeline_rgy_filter_denoise_pmd_cl'],
  ['rgy_filter_edgelevel_cl', '../QSVPipeline/rgy_filter_edgelevel.cl', 'QSVPipeline_rgy_filter_edgelevel_cl'],
  ['rgy_filter_mpdecimate_cl', '../QSVPipeline/rgy_filter_mpdecimate.cl', 'QSVPipeline_rgy_filter_mpdecimate_cl'],
  ['rgy_filter_nnedi_common_cl', '../QSVPipeline/rgy_filter_nnedi_common.cl', 'QSVPipeline_rgy_filter_nnedi_common_cl'],
  ['rgy_filter_nnedi_k0_cl', '../QSVPipeline/rgy_filter_nnedi_k0.cl', 'QSVPipeline_rgy_filter_nnedi_k0_cl'],
  ['rgy_filter_nnedi_k1_cl', '../QSVPipeline/rgy_filter_nnedi_k1.cl', 'QSVPipeline_rgy_filter_nnedi_k1_cl'],
  ['rgy_filter_overlay_cl', '../QSVPipeline/rgy_filter_overlay.cl', 'QSVPipeline_rgy_filter_overlay_cl'],
  ['rgy_filter_pad_cl', '../QSVPipeline/rgy_filter_pad.cl', 'QSVPipeline_rgy_filter_pad_cl'],
  ['rgy_filter_resize_cl', '../QSVPipeline/rgy_filter_resize.cl', 'QSVPipeline_rgy_filter_resize_cl'],
  ['rgy_filter_ssim_cl', '../QSVPipeline/rgy_filter_ssim.cl', 'QSVPipeline_rgy_filter_ssim_cl'],
  ['rgy_filter_smooth_cl', '../QSVPipeline/rgy_filter_smooth.cl', 'QSVPipeline_rgy_filter_smooth_cl'],
  ['rgy_filter_subburn_cl', '../QSVPipeline/rgy_filter_subburn.cl', 'QSVPipeline_rgy_filter_subburn_cl'],
  ['rgy_filter_transform_cl', '../QSVPipeline/rgy_filter_transform.cl', 'QSVPipeline_rgy_filter_transform_cl'],
  ['rgy_filter_tweak_cl', '../QSVPipeline/rgy_filter_tweak.cl', 'QSVPipeline_rgy_filter_tweak_cl'],
  ['rgy_filter_unsharp_cl', '../QSVPipeline/rgy_filter_unsharp.cl', 'QSVPipeline_rgy_filter_unsharp_cl'],
  ['rgy_filter_warpsharp_cl', '../QSVPipeline/rgy_filter_warpsharp.cl', 'QSVPipeline_rgy_filter_warpsharp_cl'],
  ['rgy_filter_yadif_cl', '../QSVPipeline/rgy_filter_yadif.cl', 'QSVPipeline_rgy_filter_yadif_cl'],
]

resource_objects = []
foreach spec : resource_specs
  resource_objects += custom_target('embed_' + spec[0],
    input: spec[1],
    output: spec[0] + '.o',
    command: ['bash', embed_script, '@INPUT@', '@OUTPUT@', spec[2]],
  )
endforeach
