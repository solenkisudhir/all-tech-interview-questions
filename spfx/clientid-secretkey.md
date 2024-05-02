To access data from an external API in SharePoint Framework (SPFx) using PnPjs with client credentials (client ID and client secret), you typically need to use the `SPFetchClient` class provided by PnPjs. Here's an example of how you can do this:

1. **Install PnPjs**:

Make sure you have installed PnPjs in your SPFx project. If not, you can install it using npm:

```bash
npm install @pnp/common @pnp/graph-commonjs @pnp/logging @pnp/odata @pnp/sp-commonjs @pnp/sp-utilities @pnp/sp-clientsvc
```

2. **Create a Service Class to Make API Requests**:

```typescript
import { sp } from '@pnp/sp';
import { SPFetchClient } from '@pnp/nodejs';

export default class ExternalAPIService {
  private readonly apiUrl: string = 'https://api.example.com/data';

  constructor(private readonly clientId: string, private readonly clientSecret: string) {
    sp.setup({
      sp: {
        fetchClientFactory: () => {
          return new SPFetchClient(clientId, clientSecret);
        },
      },
    });
  }

  public async fetchData(): Promise<any> {
    try {
      return await sp.web
        .getFileByServerRelativeUrl(this.apiUrl)
        .getJSON();
    } catch (error) {
      console.error('Error fetching data:', error);
      throw error;
    }
  }
}
```

3. **Use the Service in Your SPFx Web Part or Extension**:

```typescript
import * as React from 'react';
import ExternalAPIService from './ExternalAPIService';

export interface IExternalAPIWebPartProps {}

export default class ExternalAPIWebPart extends React.Component<IExternalAPIWebPartProps, any> {
  private externalAPIService: ExternalAPIService;

  constructor(props: IExternalAPIWebPartProps) {
    super(props);
    this.externalAPIService = new ExternalAPIService('your_client_id', 'your_client_secret');
    this.state = {
      data: null,
      loading: false,
      error: null,
    };
  }

  public componentDidMount(): void {
    this.fetchDataFromAPI();
  }

  private async fetchDataFromAPI(): Promise<void> {
    this.setState({ loading: true });
    try {
      const data = await this.externalAPIService.fetchData();
      this.setState({ data, loading: false });
    } catch (error) {
      this.setState({ error, loading: false });
    }
  }

  public render(): React.ReactElement<IExternalAPIWebPartProps> {
    const { data, loading, error } = this.state;

    if (loading) {
      return <div>Loading...</div>;
    }

    if (error) {
      return <div>Error: {error.message}</div>;
    }

    return (
      <div>
        {/* Render data here */}
      </div>
    );
  }
}
```

4. **Register Your Web Part or Extension in the Manifest File**:

Ensure that you have registered your web part or extension in the manifest file (`manifest.json`) and assigned the necessary permissions for accessing external APIs in the SharePoint app manifest.

This example demonstrates how to access data from an external API using PnPjs with client credentials (client ID and client secret) in SharePoint Framework. Make sure to replace `'your_client_id'` and `'your_client_secret'` with your actual client ID and client secret values.
