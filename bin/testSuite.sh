#work-group size test case
bin/tripletBuilder --config configs/bigOne.local.conf --testSuite configs/workGroupSize.test --silent

#behavior with tracks test case
bin/tripletBuilder --config configs/bigOne.local.conf --testSuite configs/tracks.test --silent
bin/tripletBuilder --config configs/bigOne.halfLocal.conf --testSuite configs/tracks.test --silent
bin/tripletBuilder --config configs/bigOne.noLocal.conf --testSuite configs/tracks.test --silent

#behavior with events test case
bin/tripletBuilder --config configs/bigOne.local.conf --testSuite configs/events.100.test --silent
bin/tripletBuilder --config configs/bigOne.local.conf --testSuite configs/events.1000.test --silent

#multiple mu sample
bin/tripletBuilder --config configs/multipleMu.local.conf --silent
bin/tripletBuilder --config configs/multipleMu.halfLocal.conf --silent
bin/tripletBuilder --config configs/multipleMu.noLocal.conf --silent

