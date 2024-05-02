In React, "HOC" stands for Higher-Order Component. Higher-Order Components are a pattern in React where a component takes another component as an argument and returns a new component. This pattern is commonly used for code reuse, logic abstraction, and to enhance the functionality of existing components.

Here's a breakdown of how Higher-Order Components work in React:

1. **Component Composition**:
   - In React, components are the building blocks of UI. They encapsulate reusable pieces of UI logic.
   - Higher-Order Components allow you to compose components by wrapping them with other components.

2. **Function that Returns a Component**:
   - A Higher-Order Component is a function that takes a component as an argument and returns a new component.
   - Inside the Higher-Order Component, you can add additional props, manipulate the component's behavior, or access its state.

3. **Usage**:
   - Higher-Order Components are typically used to share behavior between components.
   - They can be used for tasks like authentication, logging, tracking, or handling side effects.

Here's a simple example of a Higher-Order Component in React:

```jsx
// Higher-Order Component
const withLogger = (WrappedComponent) => {
  return class extends React.Component {
    componentDidMount() {
      console.log(`Component ${WrappedComponent.name} mounted`);
    }
    
    render() {
      return <WrappedComponent {...this.props} />;
    }
  };
};

// Example Component
class MyComponent extends React.Component {
  render() {
    return <div>Hello, World!</div>;
  }
}

// Enhance MyComponent with the withLogger HOC
const MyEnhancedComponent = withLogger(MyComponent);

// Usage
ReactDOM.render(<MyEnhancedComponent />, document.getElementById('root'));
```

In this example:
- `withLogger` is a Higher-Order Component that takes a component as an argument and returns a new component.
- `MyComponent` is a simple React component.
- `MyEnhancedComponent` is the result of enhancing `MyComponent` with the `withLogger` HOC.
- When `MyEnhancedComponent` is rendered, it logs a message to the console indicating that the component has mounted.

Higher-Order Components are a powerful pattern in React that promotes code reuse and separation of concerns. They are widely used in React libraries and frameworks for various purposes.
