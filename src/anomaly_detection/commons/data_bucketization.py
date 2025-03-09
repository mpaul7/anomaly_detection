
import numpy as np
import pandas as pd
from pathlib import Path

from anomaly_detection.entity.config_entity import DataTransformationConfig
from anomaly_detection.utils.common import read_file

def bucketize_data(params, df):

    df = df[(df['sport'] == 53) | (df['dport'] == 53)]
    df['pkt_flow_fwd_count'] = np.where(df['pkt_fwd_count'] > 0, 1, 0)
    df['pkt_flow_bwd_count'] = np.where(df['pkt_bwd_count'] > 0, 1, 0)

    df['pl_flow_fwd_count'] = np.where(df['pl_fwd_count'] > 0, 1, 0)
    df['pl_flow_bwd_count'] = np.where(df['pl_bwd_count'] > 0, 1, 0)

    # df['pkt_per_sec'] = (df['pkt_fwd_count'] + df['pkt_bwd_count']) / ((df['last_timestamp'] - df['first_timestamp']) / 1_000_000) # pps -ok
    # df['pkt_bytes_per_sec'] = (df['pkt_len_fwd_total'] + df['pkt_len_bwd_total']) / ((df['last_timestamp'] - df['first_timestamp']) / 1_000_000) # bps

    # df['pl_per_sec'] = (df['pl_fwd_count'] + df['pl_bwd_count']) / df['total_time']
    # df['pl_bytes_per_sec'] = (df['pl_len_fwd_total'] + df['pl_len_bwd_total']) / df['total_time']

    # df['pkt_count_ratio'] = np.where(df['pkt_fwd_count'] != 0, df['pkt_bwd_count'] / df['pkt_fwd_count'], np.nan)
    df['pkt_count_ratio'] = df['pkt_bwd_count'] / (df['pkt_fwd_count'] + 1)
    df['pkt_bytes_ratio'] = np.where(df['pkt_len_fwd_total'] != 0, df['pkt_len_bwd_total'] / df['pkt_len_fwd_total'], np.nan)

    # df['pl_count_ratio'] = np.where(df['pl_fwd_count'] != 0, df['pl_bwd_count'] / df['pl_fwd_count'], np.nan)
    df['pl_count_ratio'] = df['pl_bwd_count'] / (df['pl_fwd_count'] + 1)
    df['pl_bytes_ratio'] = np.where(df['pl_len_fwd_total'] != 0, df['pl_len_bwd_total'] / df['pl_len_fwd_total'], np.nan)   

    # Step 1: Convert `first_timestamp` to a datetime format if it's not already
    df['first_timestamp'] = (df['first_timestamp'] // 1_000_000).astype(int)
    df = df.sort_values(by='first_timestamp', ascending=True)
    df['time_difference_seconds'] = df['first_timestamp'] - df['first_timestamp'].iloc[0]

    max_time = df['time_difference_seconds'].max()
    bins = list(range(0, int(max_time) + params.settings.bucket_size, params.settings.bucket_size))

    # Use pd.cut to assign each row to a bucket based on the time difference in seconds
    df['bucket'] = pd.cut(
        df['time_difference_seconds'], 
        bins=bins, 
        labels=range(len(bins)-1),
        right=True
    )
    output_col = ['sip', 'sport', 'dip', 'dport', 'bucket', 'first_timestamp', 'time_difference_seconds',
                'pkt_bwd_count','pkt_fwd_count', 
                # 'pkt_per_sec', 
                'total_time', 'pkt_len_fwd_total', 'pkt_len_bwd_total',
                'pl_fwd_count','pl_bwd_count', 'pl_len_fwd_total', 'pl_len_bwd_total',
                'label', 'pkt_count_ratio', 'pkt_bytes_ratio', 'pl_count_ratio', 'pl_bytes_ratio',
                'pkt_flow_fwd_count', 'pkt_flow_bwd_count'
                # 'filename'
                ]
    flat_df = df.copy()
    
    def determine_agg_label(group):
        value = group[:-1]
        val = value.unique()
        if 'dns_attack' in val:
            return 'dns_attack'
        else:
            return 'dns'

    aggregated_df = df.groupby('bucket').agg(
        # Set 1 - Payload flow features
        sum_pl_fwd_count=('pl_fwd_count', 'sum'), # 1. Summation of all flows pl_fwd_count in a bucket
        sum_pl_bwd_count=('pl_bwd_count', 'sum'), # 2. Summation of all flows pl_bwd_count in a bucket
        sum_pl_fwd_bytes=('pl_len_fwd_total', 'sum'), # 3. Summation of all flows pl_len_fwd_total bytes in a bucket
        sum_pl_bwd_bytes=('pl_len_bwd_total', 'sum'), # 4. Summation of all flows pl_len_bwd_total bytes in a bucke 
        sum_pl_fwd_flow_count=('pl_flow_fwd_count', 'sum'), # 5. Payload flow count fwd in a bucket
        sum_pl_bwd_flow_count=('pl_flow_bwd_count', 'sum'), # 6. Payload flow count bwd in a bucket
        
        # sum_pkt_per_sec=('pkt_per_sec', 'sum'), # 11. Packet per second in a bucket
        # Set 2 - average of Set 1
        avg_pl_fwd_count=('pl_fwd_count', 'mean'), # 1a. Summation of all flows pl_fwd_count in a bucket
        avg_pl_bwd_count=('pl_bwd_count', 'mean'), # 2a. Summation of all flows pl_bwd_count in a bucket
        avg_pl_fwd_bytes=('pl_len_fwd_total', 'mean'), # 3a. Summation of all flows pl_len_fwd_total bytes in a bucket
        avg_pl_bwd_bytes=('pl_len_bwd_total', 'mean'), # 4a. Summation of all flows pl_len_bwd_total bytes in a bucke 
                #  'pl_flow_fwd_bwd_byte_ratio',      # 7 Ratio of payload flow bytes fwd and bwd in a bucket i.e., #3/#4

        # Set 3 - Packet flow features
        sum_pkt_fwd_count=('pkt_fwd_count', 'sum'), # 5. Summation of all flows pkt_fwd_count in a bucket
        sum_pkt_bwd_count=('pkt_bwd_count', 'sum'), # 6. Summation of all flows pkt_bwd_count in a bucket    
        sum_pkt_fwd_bytes=('pkt_len_fwd_total', 'sum'), # 7. Summation of all flows pkt_len_fwd_total bytes in a bucket
        sum_pkt_bwd_bytes=('pkt_len_bwd_total', 'sum'), # 8. Summation of all flows pkt_len_bwd_total bytes in a bucket 
        sum_pkt_fwd_flow_count=('pkt_flow_fwd_count', 'sum'), # 9. Packet flow count fwd in a bucket
        sum_pkt_bwd_flow_count=('pkt_flow_bwd_count', 'sum'), # 10. Packet flow count bwd in a bucket   

        # Set 4 - average of Set 3
        avg_pkt_fwd_count=('pkt_fwd_count', 'mean'), # 5a. average of all flows pkt_fwd_count in a bucket
        avg_pkt_bwd_count=('pkt_bwd_count', 'mean'), # 6a. average of all flows pkt_bwd_count in a bucket
        avg_pkt_fwd_bytes=('pkt_len_fwd_total', 'mean'), # 7a. average of all flows pkt_len_fwd_total bytes in a bucket
        avg_pkt_bwd_bytes=('pkt_len_bwd_total', 'mean'), # 8a. average of all flows pkt_len_bwd_total bytes in a bucket 

        
        # Row count in bucket
        bucket_size=('pl_fwd_count', 'size'), # Total number of rows in each bucket

        label=('label', determine_agg_label)  
    ).reset_index()

    # 7. Ratio of payload flow bytes fwd and bwd in a bucket i.e., #3/#4
    aggregated_df['pl_flow_fwd_bwd_byte_ratio_sum'] =  aggregated_df['sum_pl_fwd_bytes'] / aggregated_df['sum_pl_bwd_bytes']
    aggregated_df['pl_flow_fwd_bwd_byte_ratio_avg'] =  aggregated_df['avg_pl_fwd_bytes'] / aggregated_df['avg_pl_bwd_bytes']

    aggregated_df['pkt_flow_fwd_bwd_byte_ratio_sum'] =  aggregated_df['sum_pkt_fwd_bytes'] / aggregated_df['sum_pkt_bwd_bytes']
    aggregated_df['pkt_flow_fwd_bwd_byte_ratio_avg'] =  aggregated_df['avg_pkt_fwd_bytes'] / aggregated_df['avg_pkt_bwd_bytes']

    aggregated_df['pl_flow_bwd_fwd_byte_ratio'] = np.where( # 7. Ratio of payload flow bytes fwd and bwd in a bucket i.e., #4/#3
        aggregated_df['sum_pl_fwd_bytes'] != 0,
        aggregated_df['sum_pl_bwd_bytes'] / aggregated_df['sum_pl_fwd_bytes'],
        np.nan  # Assign NaN where fw_flow_count is zero
    )

    aggregated_df['pkt_flow_bwd_fwd_byte_ratio'] = np.where( # 7. Ratio of packet flow bytes fwd and bwd in a bucket i.e., #8/#7
        aggregated_df['sum_pkt_fwd_bytes'] != 0,
        aggregated_df['sum_pkt_bwd_bytes'] / aggregated_df['sum_pkt_fwd_bytes'],
        np.nan  # Assign NaN where fw_flow_count is zero
    )

    aggregated_df['pl_flow_count_ratio'] = np.where( 
        aggregated_df['sum_pl_bwd_flow_count'] != 0,
        aggregated_df['sum_pl_bwd_flow_count'] / aggregated_df['sum_pl_fwd_flow_count'],
        np.nan  # Assign NaN where fw_flow_count is zero
    )

    aggregated_df['pkt_flow_count_ratio'] = np.where( 
        aggregated_df['sum_pkt_bwd_flow_count'] != 0,
        aggregated_df['sum_pkt_bwd_flow_count'] / aggregated_df['sum_pkt_fwd_flow_count'],
        np.nan  # Assign NaN where fw_flow_count is zero
    )   

    # aggregated_df['pkt_count_ratio'] = np.where( 
    #     aggregated_df['sum_pkt_bwd_count'] != 0,
    #     aggregated_df['sum_pkt_bwd_count'] / aggregated_df['sum_pkt_fwd_count'],
    #     np.nan  # Assign NaN where fw_flow_count is zero
    # ) 
    aggregated_df['pkt_count_ratio'] = aggregated_df['sum_pkt_bwd_count'] / (aggregated_df['sum_pkt_fwd_count'] + 1)

    # aggregated_df['pl_count_ratio'] = np.where( 
    #     aggregated_df['sum_pl_bwd_count'] != 0,
    #     aggregated_df['sum_pl_bwd_count'] / aggregated_df['sum_pl_fwd_count'],
    #     np.nan  # Assign NaN where fw_flow_count is zero
    # ) 

    aggregated_df['pl_count_ratio'] = aggregated_df['sum_pl_bwd_count'] / (aggregated_df['sum_pl_fwd_count'] + 1)
    aggregated_df = aggregated_df.dropna()
    
    return aggregated_df, flat_df