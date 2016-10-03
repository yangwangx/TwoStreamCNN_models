%% spatial net
caffenet = caffe.Net('../caffe_model/spatial_cls.prototxt', '../caffe_model/spatial.caffemodel', 'train');
spatial_net_init
for i=1:23
  if numel(caffenet.layer_vec(i).params)>0
    filters=caffenet.layer_vec(i).params(1).get_data();
    biases=caffenet.layer_vec(i).params(2).get_data();
    net.layers{i}.filters=reshape(filters,size(net.layers{i}.filters));
    net.layers{i}.biases=reshape(biases,size(net.layers{i}.biases));
  end
end
load('../caffe_model/VGG_mean.mat'); % H * W * [BGR]
net.input.format='W x H x [BGR] x S';
net.input.mean=permute(image_mean,[2,1,3]);
s_net=net;
save('spatial_net.mat','s_net');

%% temporal net
caffenet = caffe.Net('../caffe_model/temporal_cls.prototxt', '../caffe_model/temporal.caffemodel', 'train');
temporal_net_init
for i=1:23
  if numel(caffenet.layer_vec(i).params)>0
    filters=caffenet.layer_vec(i).params(1).get_data();
    biases=caffenet.layer_vec(i).params(2).get_data();
    net.layers{i}.filters=reshape(filters,size(net.layers{i}.filters));
    net.layers{i}.biases=reshape(biases,size(net.layers{i}.biases));
  end
end
net.input.format='W x H x [XY^10] x S';
net.input.mean=128*ones(224,224,20,'single');
t_net=net;
save('temporal_net.mat','t_net');

