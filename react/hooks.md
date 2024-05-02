Sure, let's explore some React hooks with examples:

1. **useState() Hook**:
   - `useState` allows functional components to manage state.
   - It returns a stateful value and a function to update it.
   
```jsx
import React, { useState } from 'react';

const Counter = () => {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
};

export default Counter;
```

2. **useEffect() Hook**:
   - `useEffect` adds the ability to perform side effects in functional components.
   - It runs after every render, including the first render.
   
```jsx
import React, { useState, useEffect } from 'react';

const DataFetcher = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    // Fetch data from an API
    fetch('https://api.example.com/data')
      .then(response => response.json())
      .then(data => setData(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []); // Empty dependency array means this effect runs only once after the component mounts

  return (
    <div>
      {data ? <p>Data: {data}</p> : <p>Loading...</p>}
    </div>
  );
};

export default DataFetcher;
```

3. **useContext() Hook**:
   - `useContext` provides a way to pass data through the component tree without having to pass props manually at every level.
   
```jsx
import React, { useContext } from 'react';

// Create a context
const ThemeContext = React.createContext('light');

// Component consuming the context
const ThemeDisplay = () => {
  const theme = useContext(ThemeContext);
  return <p>Current theme: {theme}</p>;
};

// Component tree where the context is used
const App = () => {
  return (
    <ThemeContext.Provider value="dark">
      <ThemeDisplay />
    </ThemeContext.Provider>
  );
};

export default App;
```

4. **useReducer() Hook**:
   - `useReducer` is an alternative to `useState` for managing complex state logic.
   - It accepts a reducer function and an initial state, and returns the current state and a dispatch function.
   
```jsx
import React, { useReducer } from 'react';

// Reducer function
const reducer = (state, action) => {
  switch (action.type) {
    case 'increment':
      return { count: state.count + 1 };
    case 'decrement':
      return { count: state.count - 1 };
    default:
      throw new Error('Unhandled action type');
  }
};

// Component using useReducer
const Counter = () => {
  const [state, dispatch] = useReducer(reducer, { count: 0 });

  return (
    <div>
      <p>Count: {state.count}</p>
      <button onClick={() => dispatch({ type: 'increment' })}>Increment</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>Decrement</button>
    </div>
  );
};

export default Counter;
```

5. **useCallback() Hook**:
   - `useCallback` returns a memoized callback function that only changes if one of the dependencies has changed.
   - It helps in optimizing performance by avoiding unnecessary re-renders.
   
```jsx
import React, { useState, useCallback } from 'react';

const MemoizedCounter = () => {
  const [count, setCount] = useState(0);

  // Memoize the increment function
  const increment = useCallback(() => {
    setCount(prevCount => prevCount + 1);
  }, []);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>Increment</button>
    </div>
  );
};

export default MemoizedCounter;
```

These are some of the most commonly used React hooks. They provide a powerful and concise way to manage state, perform side effects, and optimize performance in functional components.
