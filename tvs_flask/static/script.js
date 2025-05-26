document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const turingScoreEl = document.getElementById('turingScore');
    const scherbiusScoreEl = document.getElementById('scherbiusScore'); // Will be set to '???'
    const maxVictoryPointsEl = document.getElementById('maxVictoryPoints');
    const winnerEl = document.getElementById('winner');
    const newGameBtn = document.getElementById('newGameBtn');
    const submitTuringActionBtn = document.getElementById('submitTuringActionBtn');
    const gameAreaEl = document.getElementById('gameArea');
    const gameOverMessageEl = document.getElementById('gameOverMessage');
    const lastRoundSummaryAreaEl = document.getElementById('lastRoundSummaryArea');
    const lastRoundInfoEl = document.getElementById('lastRoundInfo'); // For general summary text
    const lastRoundSummaryBattlesEl = document.getElementById('lastRoundSummaryBattles');
    const rewardsDisplayEl = document.getElementById('rewardsDisplay');
    const turingHandEl = document.getElementById('turingHand');
    // const encryptionGuessSlotsEl = document.getElementById('encryptionGuessSlots'); // REMOVED
    // const encryptionCodeLenEl = document.getElementById('encryptionCodeLen'); // REMOVED

    // --- Client-Side Game State ---
    let clientState = {
        initialHandForTurn: [], currentHandDisplayObjects: [],
        battlePlays: {}, 
        // encryptionGuess: [], // REMOVED
        nBattles: 0, 
        // encCodeLen: 0, // REMOVED
        draggedCard: { id: null, value: null, originType: null, originId: null, element: null }
    };

    // --- API Calls (same) ---
    async function fetchApi(endpoint, method = 'GET', body = null) {
        try {
            const options = { method };
            if (body) {
                options.headers = { 'Content-Type': 'application/json' };
                options.body = JSON.stringify(body);
            }
            const response = await fetch(endpoint, options);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }
            const state = await response.json();
            updateGlobalUI(state);
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            alert(`Error: ${error.message}`);
        }
    }

    // --- Main UI Update Function ---
    function updateGlobalUI(serverState) {
        console.log("Server state received:", serverState);
        gameAreaEl.style.display = 'block';

        turingScoreEl.textContent = serverState.turing_points;
        scherbiusScoreEl.textContent = "???"; // Hide Scherbius's score
        maxVictoryPointsEl.textContent = serverState.max_victory_points;
        
        clientState.nBattles = serverState.n_battles;
        // clientState.encCodeLen = serverState.encryption_code_len; // REMOVED
        // encryptionCodeLenEl.textContent = clientState.encCodeLen; // REMOVED

        if (serverState.current_phase === "Turing_Action") {
            clientState.initialHandForTurn = [...(serverState.turing_hand || [])]; 
            clientState.battlePlays = {};
            for(let i=0; i < clientState.nBattles; i++) clientState.battlePlays[`battle_${i}`] = [];
            // clientState.encryptionGuess = new Array(clientState.encCodeLen).fill(null); // REMOVED
        }
        
        renderAllPlayableCardAreas();
        renderRewardsAndBattleDropZones(serverState.rewards, serverState.n_battles, serverState.scherbius_observed_plays);
        
        if (serverState.last_round_summary) {
            lastRoundSummaryAreaEl.style.display = 'block';
            renderLastRoundSummary(serverState.last_round_summary);
        } else {
            lastRoundSummaryAreaEl.style.display = 'none';
        }

        if (serverState.is_game_over) {
            gameOverMessageEl.style.display = 'block';
            winnerEl.textContent = serverState.winner;
            submitTuringActionBtn.disabled = true;
        } else {
            gameOverMessageEl.style.display = 'none';
            submitTuringActionBtn.disabled = false;
        }
    }

    // --- Card Rendering and D&D ---
    function createCardElement(cardObject, originType, originId) {
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        cardDiv.textContent = cardObject.value;
        cardDiv.draggable = true;
        cardDiv.dataset.cardId = cardObject.id;
        cardDiv.dataset.cardValue = cardObject.value;
        cardDiv.dataset.originType = originType;
        cardDiv.dataset.originId = originId;

        cardDiv.addEventListener('dragstart', (e) => {
            clientState.draggedCard = { 
                id: e.target.dataset.cardId, value: parseInt(e.target.dataset.cardValue), 
                originType: e.target.dataset.originType, originId: e.target.dataset.originId,
                element: e.target
            };
            e.dataTransfer.setData('text/plain', e.target.dataset.cardId);
            setTimeout(() => e.target.classList.add('dragging'), 0);
        });
        cardDiv.addEventListener('dragend', (e) => {
            e.target.classList.remove('dragging');
            clientState.draggedCard = { id: null, value: null, originType: null, originId: null, element: null };
        });
        return cardDiv;
    }

    function findCardInArrayById(cardArray, cardId) {
        return cardArray.find(card => card && card.id === cardId);
    }
    function removeCardFromArrayById(cardArray, cardId) {
        const index = cardArray.findIndex(card => card && card.id === cardId);
        if (index > -1) return cardArray.splice(index, 1)[0];
        return null;
    }

    function renderAllPlayableCardAreas() {
        let cardsEffectivelyInHandObjects = [...clientState.initialHandForTurn];
        Object.values(clientState.battlePlays).flat().forEach(playedCardObj => {
            if (playedCardObj) {
                cardsEffectivelyInHandObjects = cardsEffectivelyInHandObjects.filter(handCardObj => handCardObj.id !== playedCardObj.id);
            }
        });
        // No encryptionGuess to filter by
        clientState.currentHandDisplayObjects = cardsEffectivelyInHandObjects;

        turingHandEl.innerHTML = '';
        clientState.currentHandDisplayObjects.forEach(cardObj => {
            turingHandEl.appendChild(createCardElement(cardObj, 'hand', 'turingHand'));
        });

        for (let i = 0; i < clientState.nBattles; i++) {
            const battleId = `battle_${i}`;
            const cardsArea = rewardsDisplayEl.querySelector(`.turing-played-cards-area[data-battle-id="${battleId}"]`);
            if (cardsArea) {
                cardsArea.innerHTML = '';
                (clientState.battlePlays[battleId] || []).forEach(cardObj => {
                    if(cardObj) cardsArea.appendChild(createCardElement(cardObj, 'battle', battleId));
                });
            }
        }
        // encryptionGuessSlotsEl.innerHTML = ''; // REMOVED
        // No rendering of encryption guess slots
    }
    
    function handleCardDrop(targetType, targetId) {
        const { id: draggedCardId } = clientState.draggedCard;
        if (!draggedCardId) return;

        let cardObjectToMove = null; 
        const originType = clientState.draggedCard.originType;
        const originId = clientState.draggedCard.originId;

        if (originType === 'hand') {
            cardObjectToMove = findCardInArrayById(clientState.initialHandForTurn, draggedCardId);
        } else if (originType === 'battle') {
            cardObjectToMove = removeCardFromArrayById(clientState.battlePlays[originId], draggedCardId);
        } 
        // No 'guess' originType

        if (!cardObjectToMove) {
            console.error("Dragged card object not found in origin!", clientState.draggedCard);
            renderAllPlayableCardAreas();
            return;
        }

        if (targetType === 'hand') {
            // Implicitly returned to hand pool
        } else if (targetType === 'battle') {
            if (!clientState.battlePlays[targetId]) clientState.battlePlays[targetId] = [];
            if (!findCardInArrayById(clientState.battlePlays[targetId], cardObjectToMove.id)) {
                 clientState.battlePlays[targetId].push(cardObjectToMove);
            }
        }
        // No 'guess' targetType
        
        renderAllPlayableCardAreas();
    }

    function addDropListeners(element, targetType, targetId) {
        element.addEventListener('dragover', e => { e.preventDefault(); element.classList.add('drag-over'); });
        element.addEventListener('dragleave', e => { element.classList.remove('drag-over'); });
        element.addEventListener('drop', e => {
            e.preventDefault();
            element.classList.remove('drag-over');
            handleCardDrop(targetType, targetId);
        });
    }
    addDropListeners(turingHandEl, 'hand', 'turingHand');

    function renderRewardsAndBattleDropZones(rewardsData, numBattles, scherbiusObservedPlays) {
        rewardsDisplayEl.innerHTML = '';
        for (let i = 0; i < numBattles; i++) {
            const battleId = `battle_${i}`;
            const battleDiv = document.createElement('div');
            battleDiv.classList.add('battle-item');
            
            let rewardInfoHTML = `<div class="reward-info"><h4>Battle ${i}</h4>`;
            const vpReward = rewardsData.vp_rewards[i];
            const cardRewardArray = rewardsData.card_rewards[i];
            if (vpReward > 0) rewardInfoHTML += `<p>VP Reward: ${vpReward}</p>`;
            if (cardRewardArray && cardRewardArray.length > 0) rewardInfoHTML += `<p>Card Rewards: ${cardRewardArray.join(', ')}</p>`;
            if (vpReward === 0 && (!cardRewardArray || cardRewardArray.length === 0)) rewardInfoHTML += `<p>No direct reward</p>`;
            rewardInfoHTML += `</div>`;

            let scherbiusInfoHTML = `<div class="scherbius-observed-info">`;
            const scherbiusPlayForBattle = (scherbiusObservedPlays && scherbiusObservedPlays[i]) ? scherbiusObservedPlays[i] : [];
            if (scherbiusPlayForBattle.length > 0) {
                scherbiusInfoHTML += `<p>Scherbius played: ${scherbiusPlayForBattle.join(', ')}</p>`;
            } else {
                scherbiusInfoHTML += `<p>Scherbius played: None</p>`;
            }
            scherbiusInfoHTML += `</div>`;

            const dropAreaHTML = `<div class="turing-played-cards-area" data-battle-id="${battleId}"></div>`;
            battleDiv.innerHTML = rewardInfoHTML + scherbiusInfoHTML + dropAreaHTML;
            rewardsDisplayEl.appendChild(battleDiv);

            const dropArea = battleDiv.querySelector('.turing-played-cards-area');
            addDropListeners(dropArea, 'battle', battleId);
        }
    }
    
    function renderLastRoundSummary(summary) {
        let infoText = `Points Gained This Round - Turing: ${summary.turing_points_gained_in_round}, Scherbius: ${summary.scherbius_points_gained_in_round}. `;
        lastRoundInfoEl.innerHTML = infoText;

        lastRoundSummaryBattlesEl.innerHTML = '';
        summary.battle_details.forEach(battle => {
            const reportDiv = document.createElement('div');
            reportDiv.classList.add('summary-battle-report');
            reportDiv.innerHTML = `
                <h5>Battle ${battle.battle_id}</h5>
                <p>Turing Played: <span class="cards-list">${battle.turing_played.length > 0 ? battle.turing_played.join(', ') : 'None'}</span></p>
                <p>Scherbius Committed: <span class="cards-list">${battle.scherbius_committed.length > 0 ? battle.scherbius_committed.join(', ') : 'None'}</span></p>
            `;
            lastRoundSummaryBattlesEl.appendChild(reportDiv);
        });
    }

    // --- Event Listeners ---
    newGameBtn.addEventListener('click', () => {
        lastRoundSummaryAreaEl.style.display = 'none';
        gameOverMessageEl.style.display = 'none';
        fetchApi('/new_game', 'POST');
    });

    submitTuringActionBtn.addEventListener('click', () => {
        const finalTuringStrategyValues = [];
        for (let i = 0; i < clientState.nBattles; i++) {
            finalTuringStrategyValues.push(
                (clientState.battlePlays[`battle_${i}`] || []).map(cardObj => cardObj.value)
            );
        }
        
        // finalTuringGuessesValues is no longer needed
        const payload = { 
            turing_strategy: finalTuringStrategyValues,
            // turing_guesses: [] // Send empty or omit if backend handles missing key
        };
        console.log("Submitting to backend (values):", payload);
        fetchApi('/submit_turing_action', 'POST', payload);
    });

    // --- Initial Setup ---
    gameAreaEl.style.display = 'none';
    gameOverMessageEl.style.display = 'none';
    lastRoundSummaryAreaEl.style.display = 'none';
});