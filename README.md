cartpole-reinforce-lang-comparison
==

A naïve (policy gradient with REINFORCE) RL solution to cart-pole balancing implemented in several languages and frameworks.

My motivaition is to play a little bit with (https://github.com/flashlight/flashlight)[flashlight], see how it compares to (https://pytorch.org/[pytorch].

This is WIP.

Current status:
* Implementations produce similar results (as intended).
* Flashlight doesn't require much more code than Pytorch, but compile cycle is painful.
* Pytorch is about twice as fast in the systems I tried (surprise).
