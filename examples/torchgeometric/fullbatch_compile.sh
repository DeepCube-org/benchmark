echo Latency with batch size = 1

python metrics.py --model_name GIN       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100

python metrics.py --model_name GIN       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100

python metrics.py --model_name GIN       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 1 --throughput_batch_size 0 --compile 1 --repetitions 100

echo Throughput with batch size = 32

python metrics.py --model_name GIN       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100

python metrics.py --model_name GIN       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100

python metrics.py --model_name GIN       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 0 --throughput_batch_size 32 --compile 1 --repetitions 100

echo Latency with batch size = 32

python metrics.py --model_name GIN       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 1 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100

python metrics.py --model_name GIN       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 2 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100

python metrics.py --model_name GIN       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GraphSAGE --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GCN       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GAT       --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
python metrics.py --model_name GATv2     --n_layers 3 --num_nodes 10000 --num_edges 200000 --latency_batch_size 32 --throughput_batch_size 0 --compile 1 --repetitions 100
