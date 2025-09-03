export interface Demo{
  path: string,
  name: string,
  value: string
}

export const demos: {[key: string]: Demo} = {
  // "gaze": { path: '/gaze', name: 'Gaze', value: "gaze" },
  // "gaze-v2": { path: '/gaze-v2', name: 'Gaze-V2', value: "gaze-v2" },
  "gazev3": { path: '/gazev3', name: 'Gaze-V3', value: "gazev3" },
  "helmet": { path: '/helmet', name: 'Helmet', value: "helmet" }
}
