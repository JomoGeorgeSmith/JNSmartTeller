import React, { useState, useEffect } from 'react';
import './ChatBoxStyle.css';

function App() {
  const [accountNo, setAccountNo] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [transactionAmount, setTransactionAmount] = useState('');
  const [stage, setStage] = useState('search'); // 'search', 'list', or 'form'

  const handleSearch = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ account_no: accountNo }),
      });
      const data = await response.json();
      setPredictions(data);
      setStage('list');
    } catch (error) {
      console.error('Error fetching prediction:', error);
    }
  };

  const handleRowClick = (transaction) => {
    setSelectedTransaction(transaction);
    setTransactionAmount('');
    setStage('form');
  };

  const handleSend = () => {
    console.log('Edited Transaction:', {
      ...selectedTransaction,
      amount: transactionAmount,
    });
    alert('Transaction Complete!');
    setStage('list');
  };

  useEffect(() => {
    if (stage === 'search') {
      const rippleContainer = document.querySelector('.ripple-container');
      const colors = ['#f5b70f', '#089ccc', '#cc0505', '#e7e0c9', '#9bcc31'];
      const staticCircles = Array.from({ length: 20 }, () => ({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
      }));

      const createRipple = (x, y, color) => {
        const ripple = document.createElement('div');
        ripple.className = 'ripple';
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;
        ripple.style.width = `10px`;
        ripple.style.height = `10px`;
        ripple.style.borderColor = color;
        rippleContainer.appendChild(ripple);

        setTimeout(() => {
          ripple.remove();
        }, 4000);
      };

      const generateRipples = () => {
        for (let i = 0; i < 3; i++) {
          const { x, y } = staticCircles[Math.floor(Math.random() * staticCircles.length)];
          const randomColor = colors[Math.floor(Math.random() * colors.length)];
          createRipple(x, y, randomColor);
        }
      };

      const interval = setInterval(generateRipples, 500);
      return () => clearInterval(interval);
    }
  }, [stage]);
  return (
    <div className={`chatbox stage-${stage}`}>
      {stage === 'search' && (
        <div className="initial-screen">
          <div className="ripple-container"></div>
          <h1 className="welcome-title">Welcome To JN Smart Teller</h1>
          <input
            type="text"
            className="initial-input"
            placeholder="Account Number"
            value={accountNo}
            onChange={(e) => setAccountNo(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button className="microphone-button" onClick={handleSearch}>
            Search
          </button>
        </div>
      )}
  
    {stage === 'list' && (
    <div className="transaction-list">
      <h2 className="welcome-title">Top Predicted Transactions</h2>
      {Array(3) // Limiting to top 3 predictions
        .fill()
        .map((_, index) => (
          <div
            key={index}
            className="transaction-row"
            onClick={() =>
              handleRowClick({
                transactionType: predictions['Transaction Types'][index].label,
                transactionTypeProbability: predictions['Transaction Types'][index].probability,
                currency: predictions['Transaction Currencies'][index].label,
                currencyProbability: predictions['Transaction Currencies'][index].probability,
                branch: predictions['Transaction Branches'][index].label,
                branchProbability: predictions['Transaction Branches'][index].probability,
              })
            }
          >
            <span>{predictions['Transaction Types'][index].label}</span>
            <span>{predictions['Transaction Currencies'][index].label}</span>
            <span>{predictions['Transaction Branches'][index].label}</span>
            <span>
              {(
                (predictions['Transaction Types'][index].probability +
                  predictions['Transaction Currencies'][index].probability +
                  predictions['Transaction Branches'][index].probability) /
                3
              ).toFixed(2)}
              %
            </span>
          </div>
        ))}
    </div>
  )}

  
{stage === 'form' && selectedTransaction && (
  <div className="form-screen">
    <h2 className="welcome-title">Edit Transaction</h2>
    <div className="form-group">
      <label>Transaction Type:</label>
      <input
        type="text"
        value={selectedTransaction.transactionType}
        onChange={(e) =>
          setSelectedTransaction({ ...selectedTransaction, transactionType: e.target.value })
        }
      />
    </div>
    <div className="form-group">
      <label>Transaction Currency:</label>
      <input
        type="text"
        value={selectedTransaction.currency}
        onChange={(e) =>
          setSelectedTransaction({ ...selectedTransaction, currency: e.target.value })
        }
      />
    </div>
    <div className="form-group">
      <label>Transaction Branch:</label>
      <input
        type="text"
        value={selectedTransaction.branch}
        onChange={(e) =>
          setSelectedTransaction({ ...selectedTransaction, branch: e.target.value })
        }
      />
    </div>
    <div className="form-group">
      <label>Transaction Amount:</label>
      <input
        type="text"
        value={transactionAmount}
        onChange={(e) => setTransactionAmount(e.target.value)}
      />
    </div>
    <button className="microphone-button" onClick={handleSend}>
      Save
    </button>
  </div>
)}

    </div>
  );
  
}

export default App;

