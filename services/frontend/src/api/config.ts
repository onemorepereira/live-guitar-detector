/** Fetches browser-consumable gateway config (currently just iceServers). */

export interface IceServer {
  urls: string | string[];
  username?: string;
  credential?: string;
}

export interface GatewayConfig {
  iceServers: IceServer[];
}

export async function fetchConfig(baseUrl = ""): Promise<GatewayConfig> {
  const resp = await fetch(`${baseUrl}/api/config`);
  if (!resp.ok) {
    throw new Error(`config fetch failed (${resp.status})`);
  }
  return (await resp.json()) as GatewayConfig;
}
