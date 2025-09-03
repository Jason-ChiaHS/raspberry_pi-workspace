<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import type { WithElementRef } from 'bits-ui';
	import { cn } from '$lib/utils.js';
	import { cubicOut } from 'svelte/easing'; // Import an easing function

	let {
		transitionSlideRight = false,
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLTableRowElement>> & {
		transitionSlideRight?: boolean;
	} = $props();

	// Custom transition function: slideRight
	function slideRight(
		node: HTMLElement,
		{ delay = 0, duration = 400, easing = cubicOut, x = 100 } = {}
	) {
		const style = getComputedStyle(node);
		const opacity = +style.opacity; // Get initial opacity for a subtle fade effect

		return {
			delay,
			duration,
			easing,
			css: (t: number) => {
				// 't' is the progress of the transition, from 0 to 1 for 'in' and 1 to 0 for 'out'.
				// For 'in' (coming from right to left):
				//   When t=0 (start), we want translateX(100%)
				//   When t=1 (end), we want translateX(0%)
				// So, (1 - t) * x gives: (1-0)*100 = 100; (1-1)*100 = 0
				const transformX = (1 - t) * x;

				// For 'out' (going from left to right):
				//   When t=1 (start), we want translateX(0%)
				//   When t=0 (end), we want translateX(100%)
				// So, (1 - t) * x still works: (1-1)*100 = 0; (1-0)*100 = 100

				return `
          transform: translateX(${transformX}%);
          opacity: ${t * opacity}; /* Optional: Fade in/out along with slide */
        `;
			}
		};
	}
</script>

{#if transitionSlideRight}
	<tr
		bind:this={ref}
		class={cn(
			'hover:bg-muted/50 data-[state=selected]:bg-muted border-b transition-colors',
			className
		)}
		transition:slideRight
		{...restProps}
	>
		{@render children?.()}
	</tr>
{:else}
	<tr
		bind:this={ref}
		class={cn(
			'hover:bg-muted/50 data-[state=selected]:bg-muted border-b transition-colors',
			className
		)}
		{...restProps}
	>
		{@render children?.()}
	</tr>
{/if}
