
> Reference For ML: https://github.com/patrickloeber/MLfromscratch

> docker compose down --volumes

> docker compose up -d

> docker exec -it kafka_docker-kafka-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list

> docker exec -it kafka_docker-kafka-1 /opt/kafka/bin/kafka-topics.sh --create --zookeeper zookeeper:2181 --replication-factor 1 --partitions 1 --topic winddata
Created topic winddata.

> docker exec -it kafka_docker-kafka-1 /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
winddata

> spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 producer.py

> docker exec -it kafka_docker-kafka-1 /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic winddata --from-beginning