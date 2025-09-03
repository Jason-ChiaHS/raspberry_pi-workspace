import { page } from '$app/state';

export function getPaths(): string[] {
	let paths = page.url.pathname.split('/');
	paths.shift();
	return paths;
}
