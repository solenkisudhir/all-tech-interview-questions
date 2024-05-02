Custom hooks in React are a powerful way to reuse logic across different components. They allow you to extract stateful logic from components into reusable functions, promoting code reuse and separation of concerns. Here's how you can create and use custom hooks in React:

1. **Creating a Custom Hook**:
   - A custom hook is just a regular JavaScript function that uses React hooks internally.
   - Custom hooks must start with the word "use" to follow the React convention.

   ```jsx
   import { useState, useEffect } from 'react';

   // Custom hook to fetch data from an API
   const useFetchData = (url) => {
     const [data, setData] = useState(null);
     const [loading, setLoading] = useState(true);
     const [error, setError] = useState(null);

     useEffect(() => {
       const fetchData = async () => {
         try {
           const response = await fetch(url);
           const data = await response.json();
           setData(data);
           setLoading(false);
         } catch (error) {
           setError(error);
           setLoading(false);
         }
       };

       fetchData();
     }, [url]);

     return { data, loading, error };
   };

   export default useFetchData;
   ```

2. **Using the Custom Hook**:
   - You can use the custom hook in any function component to reuse the logic defined in the hook.

   ```jsx
   import React from 'react';
   import useFetchData from './useFetchData';

   const MyComponent = () => {
     const { data, loading, error } = useFetchData('https://api.example.com/data');

     if (loading) {
       return <div>Loading...</div>;
     }

     if (error) {
       return <div>Error: {error.message}</div>;
     }

     return (
       <div>
         {/* Render UI using data */}
       </div>
     );
   };

   export default MyComponent;
   ```

3. **Custom Hook Composition**:
   - You can compose custom hooks to create more complex functionality or combine multiple hooks together.

   ```jsx
   import React from 'react';
   import useFetchData from './useFetchData';

   const MyComponent = () => {
     const { data: userData, loading: userLoading, error: userError } = useFetchData('https://api.example.com/user');
     const { data: postsData, loading: postsLoading, error: postsError } = useFetchData('https://api.example.com/posts');

     // Combine data from different hooks or perform additional logic

     return (
       <div>
         {/* Render UI */}
       </div>
     );
   };

   export default MyComponent;
   ```

Custom hooks allow you to encapsulate and reuse complex logic across multiple components, making your code more modular, readable, and maintainable. They enable you to abstract away implementation details and focus on building reusable functionality that can be easily shared and reused across your React application.
