ps -efww|grep -w 'config=sc2'|grep -v grep|cut -c 9-16|xargs kill -9