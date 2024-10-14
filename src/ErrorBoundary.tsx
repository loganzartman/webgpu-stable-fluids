import React from "react";

type Props = {
  children: React.ReactNode;
  onError?: (error: Error) => void;
};

type State = {
  error: Error | null;
};

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error) {
    this.props.onError?.(error);
    this.setState({ error });
  }

  render() {
    return this.state.error
      ? !this.props.onError && <div>An error occurred.</div>
      : this.props.children;
  }
}
