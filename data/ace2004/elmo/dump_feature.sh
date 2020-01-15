file=$1

allennlp elmo sentences/${file}_sentences elmo_${file}.hdf5 --all
