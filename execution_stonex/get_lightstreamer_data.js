const ls = require('lightstreamer-client-node');
const fs = require('fs');

function getConstants() {
  const env = JSON.parse(fs.readFileSync('../files/env.json', 'utf8'));
  const tmp = JSON.parse(fs.readFileSync('../files/tmp.json', 'utf8'));
  return [env, tmp];
}
function connectToLS(env, tmp) {
  var client = new ls.LightstreamerClient();
  client.connectionDetails.setServerAddress(serverAddress="https://push.cityindex.com/");
  client.connectionDetails.setUser(user=env['auth']['username']);
  client.connectionDetails.setPassword(password=tmp['session_id']);
  client.connectionDetails.setAdapterSet(adapterSet="STREAMINGALL");
  client.addListener({
    onStatusChange: function(newStatus) {         
      console.log('STATUS:' + newStatus);
    },
    onPropertyChange: function(newProperty) {         
      console.log('PROPERTY:' + newProperty);
    },
    onServerError: function(newError) {         
      console.log('ERROR:' + newError);
    }
  });
  client.connect();
  return client;
}

function subscribeToStream(client, target, adapter, fields) {
  var table = new ls.Subscription("MERGE",[target],fields);
  table.setDataAdapter(adapter);
  table.addListener({
    onSubscription: function() {
    console.log("SUBSCRIBED to " + target);
  },
  onUnsubscription: function() {
    console.log("UNSUBSCRIBED from " + target);
  },
  onSubscriptionError: function(code, message) {
    console.log("SUBSCRIPTION ERROR");
    console.log(typeof(code));
    console.log(typeof(message));
    console.log(code);
    console.log(message);
  },
  onItemUpdate: function(obj) {
    try {
    time_diff = 5;
    var dataObj = {};
    for (let field of fields) {
      dataObj[field] = obj.getValue(field);
    }
    dataObj['timestamp'] = Math.floor(Date.now() / 1000);
    let file_path =  `/Users/georg/OneDrive/Documents/st0nks/forex/streaming/${adapter}_${target}.txt`;
    let dataObjStr = JSON.stringify(dataObj) + '\n';
    console.log(dataObjStr);
    fs.writeFileSync(file_path,
      dataObjStr,
      {
        flag: 'a+'
      });
    } catch(err) {
      console.log(err);
    }
  }});
  client.subscribe(table);
  return client;
}

function connectAndStream(env, tmp) {
  [env, tmp] = getConstants();
  var client = connectToLS(env, tmp);
  for (const market of Object.values(env['markets'])) {
    client = subscribeToStream(client, target=`PRICE.${market}`, adapter='PRICES', ['MarketId', 'Bid', 'Offer'])
  }
  client = subscribeToStream(client, target='CLIENTACCOUNTMARGIN', adapter='CLIENTACCOUNTMARGIN', ['Cash', 'Margin', 'TradeableFunds', 'TotalMarginRequirement', 'NetEquity', 'OpenTradeEquity']);
}
setInterval(connectAndStream, 15 * 60 * 1000);
