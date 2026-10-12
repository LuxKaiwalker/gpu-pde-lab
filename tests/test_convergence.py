from gpu_pde_lab.benchmarking.convergence import spatial_convergence, temporal_convergence


def test_spatial_convergence_is_approximately_second_order():
    result = spatial_convergence(sizes=[16, 32, 64, 128], t_final=0.01)
    # average the observed orders -- any single pair can be noisy, but the
    # 5-point stencil should be unambiguously close to order 2 overall.
    avg_order = sum(result.observed_orders) / len(result.observed_orders)
    assert 1.7 < avg_order < 2.3


def test_spatial_error_shrinks_monotonically_with_h():
    result = spatial_convergence(sizes=[16, 32, 64, 128], t_final=0.01)
    errors = [p.error_linf for p in result.points]
    assert all(e2 < e1 for e1, e2 in zip(errors, errors[1:]))


def test_temporal_convergence_shrinks_before_hitting_spatial_floor():
    dt_max = (1.0 / 511) ** 2 / (4 * 0.1)
    result = temporal_convergence(dts=[dt_max * 0.8, dt_max * 0.4], n=512, t_final=0.02)
    errors = [p.error_linf for p in result.points]
    assert errors[1] < errors[0]
