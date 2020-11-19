---
tags:
  - projects
  - data-science
  - python
  - blackjack
author: Jocelyn Ong
---
Blackjack - Can we win?

{% include toc %}

# Introduction

I recently went to Las Vegas and it sparked the beginning of a project! Blackjack! Anyone (of the right age) can play, but can we win against the house?

It's essentially a game of chance, but I'm really skeptical when it comes to casinos. To be able to stay in business, they have to be in a net win position, so can we really win?

So I thought, hey I've got the tools, let's look at the numbers and see what they say.

# Project details

So there are a few parts to this project:

- How to play blackjack
  - What's the recommended strategy
- Get the data
- Exploratory analysis of the dataset
- Can we use machine learning to come up with a strategy that's better than the current recommended strategy?

# How to play blackjack

I'm sure there are lots of great websites out there teaching you how to play blackjack in LV, but I decided to use the guide on [Las Vegas How-To](http://www.lasvegas-how-to.com/blackjack.php){:target="_blank"} because that's one of the first few that pop up on google.

So a basic game of blackjack:

- Aim: Try to get as close to 21 as possible without going over
  - If you go over 21, it's a bust and you automatically lose
- You start with 2 cards
  - If you get 21 with 2 cards (Ace + a 10, J, Q or K), it's a blackjack
    - Unless the dealer also has a blackjack, you automatically win
      - Otherwise it's a draw
  - You can take as many cards as you want
- When you're done (and you didn't bust), it's the dealer's turn
  - If the dealer has anything below 17, he has to hit
  - If he has a soft 17, he has to hit
  - Anything 17 and above (apart from soft 17), he has to stand
- If you both didn't bust, then the person with the highest points wins

Notice the soft 17. In blackjack, the Ace can be 1 or 11 points. How to decide whether it's a 1 or 11? By default, it's 11 unless the 11 causes the hand to bust. If it does, then it's a 1. If the Ace is being counted as 11 points, it's considered a soft hand.

## And the recommended strategy

According to [Las Vegas How-To](http://www.lasvegas-how-to.com/blackjack.php){:target="_blank"}, in general, you want to:

- Stand on 17 or more
- If you have 12-16 points (inclusive), and the dealer is showing 16 or less, hit
  - I believe the assumption is that the dealer's closed card always has a value of 10
    - So 16 or less, means if their open card is 6 or below
- Always split 8's
- Double down on 11 if the dealer's open card is 7 or less
- If you have 11 or lower, always hit
  - It's not possible for you to bust

Some other tips (which seem to be for slightly more advanced players):

- If you have 17 points and the dealer shows 7 or higher, consider hitting
  - The chance of you busting is very high
  - But if his open card is 7 or higher, and assuming the closed card is 10, you're going to lose anyway
- If you have 12-16 points (inclusive), and the dealer is showing 2 or 3, consider hitting
- If you have a soft hand (Ace that is counted as 11 points), consider hitting to get a better hand

# Getting the data

I believe if you google "blackjack dataset", you should be able to find lots of different datasets on the web. But I wanted to create my own - we should be able to simulate a blackjack game since it's just a game of chance (like flipping a coin).

It was more difficult than I thought. And every time I changed my mind about what data I needed, I had to generate my dataset all over again.

__What we need:__

- To be able to generate a 'deck' of cards that we can shuffle and deal from
- To be able to calculate points given a hand of cards
- To be able to automate a game

__Problems/ points to consider:__

- Many casinos use multiple decks of 52 cards to play blackjack
- Do we shuffle in between hands or do we play till the 'deck' has less than a certain number of cards?
- How do we code for the number of points to attribute to an Ace?
- Can we code in the recommended strategy so that we can automate the game?

# Starting the coding process

I started this project about slightly more than a week back, and I started off in a jupyter notebook, just throwing pieces of code in as and when I thought of them. After a while, I started 'losing' things, code was getting messy and I had to go through the thought process again when I reread the code - not going to work!

So it was time for a clean up:

- Notebook for planning
  - What are we trying to find out
  - How to play blackjack
  - Strategies
  - Risks and assumptions
  - Recap of questions with more detail

- Notebook for game simulation
  - Code requirements
  - Code the deck
  - Code the points calculator
  - Code the game
  - Test that our game works correctly

- Notebook for data analysis

- Notebook for machine learning

# What's next

I'll be spending some time on this project - follow along in my [data science projects repo](https://github.com/jocelyn-ong/data-science-projects/tree/master/others/blackjack){:target="_blank"}! The notebooks are changing all the time because I'm changing my mind all the time. My dataset will also be available there so feel free to download it (or the code to generate it) - although it might change too - and play around with it.
