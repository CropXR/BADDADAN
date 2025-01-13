from pathlib import Path

from Expressions.ExpressionMatrix import AggregationMethod, \
    ExpressionMatrixTimeSeries
from helpers import get_info_from_emtab375, get_info_from_gse65046


def expr_mat_from_heat(in_path: str, agg_method: AggregationMethod, do_log2: bool, gpl_path=None):
    expr_mat_time: ExpressionMatrixTimeSeries = ExpressionMatrixTimeSeries.from_csv(
            in_path, log2_transform=do_log2, gpl_path=gpl_path)
    expr_mat_time.keep_only_samples_with_string('normal light')
    expr_mat_time.aggregation_method = agg_method
    expr_mat_time.condition_names = ['21', '32']
    expr_mat_time.column_parser = get_info_from_emtab375
    return expr_mat_time


def expr_mat_from_drought(in_file_path: str, agg_method: AggregationMethod, do_log2: bool):
    if in_file_path.endswith('csv'):
        expr_mat_time: ExpressionMatrixTimeSeries = ExpressionMatrixTimeSeries.from_csv(
            in_file_path, log2_transform=do_log2)
    else:
        expr_mat_time: ExpressionMatrixTimeSeries = ExpressionMatrixTimeSeries.from_geo_file(
            in_file_path, annotate_from_gpl=True, log2_transform=do_log2)
    expr_mat_time.column_parser = get_info_from_gse65046
    expr_mat_time.aggregation_method = agg_method
    expr_mat_time.condition_names = ['control', 'drought']
    # expr_mat_time.merge_biological_samples()
    return expr_mat_time


def expr_mat_time_factory(folder: Path,
                          expression_path: str,
                          agg_method: AggregationMethod,
                          do_log2: bool,
                          gpl_path = None
                          ) -> ExpressionMatrixTimeSeries:
    if folder.name.startswith('drought'):
        expr_mat_time = expr_mat_from_drought(
            in_file_path=expression_path,
            agg_method=agg_method,
            do_log2=do_log2)
    elif folder.name.startswith('heat'):
        expr_mat_time = expr_mat_from_heat(in_path=expression_path,
                                           agg_method=agg_method,
                                           do_log2=do_log2,
                                           gpl_path=gpl_path)
    else:
        raise NotImplementedError
    return expr_mat_time
