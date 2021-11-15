'''
use mysql as a key object store
'''

import cPickle as pickle

class MySQL:
    def __init__(self, host='localhost',user='guest', db_name='tmp'):
        import MySQLdb
        self.db = MySQLdb.connect(host=host, user=user, db=db_name)

    def __del__(self):
        self.db.close()

    def __setitem__(self, k, v):
        vp = pickle.dumps(v, protocol=-1)
        cur = self.db.cursor()
        r = cur.execute('insert into key_store values (%s, %s)', (k, vp))
        self.db.commit()
        return r


    def __getitem__(self, k):
        r = None
        cur = self.db.cursor()
        cur.execute('select * from key_store where id = (%s)', (k,))
        for row in cur.fetchall():
            assert row[0] == k
            r = pickle.loads(row[1])
            break
        return r

    def get_del(self, k):
        r = self.get(k)
        self.delete(k)
        return r


    def __delitem__(self, k):
        cur = self.db.cursor()
        r = cur.execute('delete from key_store where id = (%s)', (k,))
        self.db.commit()
        return r

    def __contains__(self, k):
        cur = self.db.cursor()
        cur.execute('select count(id) from key_store where id = (%s)', (k,))
        c = 0
        for row in cur.fetchall():
            c = row[0]
            break
        return c > 0


    def clear_all(self):
        cur = self.db.cursor()
        r = cur.execute('truncate key_store')
        self.db.commit()
        return r

'''
ipython --pdb -c 'import aitom.io.db.mysql as TIDM; TIDM.test()'
'''
def test():
    import numpy as N
    def gen_one_record():
        r = N.random.random(3)
        import pickle, uuid
        k = str(uuid.uuid4())
        return k, r

    rs = {}
    for i in range(3):
        kv = gen_one_record()
        rs[kv[0]] = kv[1]

    d = MySQL()
    for k in rs:    d.set(k, rs[k])

    print '\nhas_key()'
    for k in rs:    print k, d.has_key(k),

    print '\nget_del()'
    for k in rs:
        v = d.get_del(k)
        assert N.all(v == rs[k])
        print k, v

    print '\nhas_key()'
    for k in rs:    print k, d.has_key(k),

    print '\nclear_all()', d.clear_all()







'''

# mysql setup example,  the uuid key has a length of 36

mysql --host=localhost --user=root --password

create user guest@localhost;
create database amarokdb;
grant all privileges on tmp.* to guest@localhost;

drop user guest@localhost;

mysql --host=localhost --user=guest
use tmp;
show tables;
create table key_store (id varchar(36) not null primary key, value longblob);

describe key_store;

select * from key_store;

truncate key_store;

drop table key_store;

'''


'''

# testing commands

%reset -f
import MySQLdb

db = MySQLdb.connect(host='localhost', user='guest', db='tmp')

cur = db.cursor()

import numpy as N
r = N.random.random(3)
import pickle, uuid
rs = pickle.dumps(r, protocol=-1)
k = str(uuid.uuid4())
print k, r
er = cur.execute('insert into tmp.key_store values (%s, %s)', (k, rs))
print 'er', er
db.commit()

#cur.execute('select * from tmp.key_store')
cur.execute('select * from tmp.key_store where id = (%s)', (k,))
for row in cur.fetchall():      print row[0], pickle.loads(row[1])

db.close()

'''


'''
see also
https://github.com/sanpingz/mysql-connector
'''
