`React.memo()` is a higher-order component (HOC) provided by React that is used for optimizing functional components by memoizing them. Memoization is a technique used to cache the result of a function call based on its input and return the cached result when the same input occurs again.

When you wrap a component with `React.memo()`, React memoizes the component, preventing unnecessary re-renders if its props haven't changed. It's particularly useful for optimizing performance in scenarios where a component's props are frequently changing but the component itself doesn't depend on those props to render.

Here's how you can use `React.memo()`:

```jsx
import React from 'react';

const MyComponent = React.memo(({ prop1, prop2 }) => {
  // Component logic here
  return (
    <div>
      <p>Prop 1: {prop1}</p>
      <p>Prop 2: {prop2}</p>
    </div>
  );
});

export default MyComponent;
```

In this example:
- `MyComponent` is a functional component that takes two props, `prop1` and `prop2`.
- It is wrapped with `React.memo()` to memoize the component.
- If `prop1` and `prop2` remain the same between renders, React will reuse the memoized version of the component without re-rendering it.

It's important to note that `React.memo()` performs a shallow comparison of props by default. If your component's props are complex objects, you might need to use custom comparison logic by passing a second argument to `React.memo()` as a comparison function.

Here's an example of using a custom comparison function:

```jsx
const MyComponent = React.memo(({ prop1, prop2 }) => {
  // Component logic here
  return (
    <div>
      <p>Prop 1: {prop1}</p>
      <p>Prop 2: {prop2}</p>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison logic
  return prevProps.prop1 === nextProps.prop1 && prevProps.prop2 === nextProps.prop2;
});
```

This custom comparison function compares the `prop1` and `prop2` values of the previous props (`prevProps`) with the next props (`nextProps`). If both props are the same, React will reuse the memoized version of the component without re-rendering it. Otherwise, it will re-render the component with the new props.
