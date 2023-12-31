$(document).ready(function() {
  var steps = [];
  var index = -1;
  const logElement = document.getElementById("log");
  var play = false;

  var socket = io({
      reconnection: false,
  });

  $("#play").click(function() {
      play = true;
      var loop = setInterval(function() {
          if (play) {
              if (index < steps.length - 1) {
                  index++;
                  drawGrid(steps[index]);
              } else {
                  play = false;
                  clearInterval(loop);
              }
          } else {
              clearInterval(loop);
          }
      }, $("#time").val());

  });

  $("#stop").click(function() {
      play = false;
  });

  $("#reset").click(function() {
      index = 0;
      drawGrid(steps[index]);
  });

  connect_handler = () => {
      socket = io("ws://" + $("#hostname")[0].value + ":" + $("#port")[0].value + "", {
          reconnection: false,
      });

      socket.on("connect_error", (err) => {
          $("#connect").addClass("connection_error");


      });

      socket.on("connect", () => {
          $("#connect").removeClass("connection_error");
          socket.emit("identify", JSON.stringify({
              "identifier": "__GUI__" + Date.now()
          }));
          $("#status")[0].innerHTML = 'Connected';
          $("#status")[0].style = 'color:green';
          $("#connect").unbind();
      });

      socket.on("play", (...args) => {
          json = JSON.parse(args[0]);
          if (!json.rep) json = JSON.parse(json);
          if (json.rep && json.rep.env) {
              current_step = json.step + "/" + json.max_step
              convertedGrid = convertToGridData({
                  "board": json.rep.env,
                  "scores": json.scores,
                  "next_player": json.next_player,
                  "players": json.players,
                  "current_step": current_step
              });
              console.log(json);
              steps.push(convertedGrid);
              index = steps.length - 1;
              drawGrid(convertedGrid);
          }
      });

      socket.on("ActionNotPermitted", (...args) => {
          // TODO: Display message to user
          $("#error").css("opacity", "1");
          setTimeout(function() {
              $("#error").css("opacity", "0");
          }, 2000);
      })

      socket.on("disconnect", (...args) => {
          // set display block to img of id #img
          $("#status")[0].innerHTML = 'Disconnected';
          $("#status")[0].style = 'color:red';
          $("#connect").click(connect_handler);
          //$("#img").css("display", "block");
      });

      socket.on("done", (...args) => {
          score = JSON.parse(args[0])
          winner_id = ""
          winner_score = -10

          for (p in score) {
              if (score[p] > winner_score) {
                  winner_id = p
                  winner_score = score[p]
              }
          }

          loser_score = winner_score

          for (p in score) {
              if (score[p] != loser_score) {
                  loser_score = score[p]
                  break;
              }
          }

          winner_score = -winner_score
          loser_score = -loser_score

          if (winner_score == loser_score) {
              text = "Match nul ! Scores : " + winner_score + " - " + loser_score
          } else {
              winner_name = ""
              winner_color = ""
              for (p in steps[steps.length - 1].players) {
                  if (steps[steps.length - 1].players[p].id == winner_id) {
                      winner_name = steps[steps.length - 1].players[p].name
                      winner_color = steps[steps.length - 1].players[p].piece_type
                  }
              }
              text = "Le joueur " + winner_name + " (<span class=\"winner_indicator " + winner_color + "\"></span>) est vainqueur avec " + loser_score + " points contre " + winner_score
          }
          $("#pop_up_container").css("display", "flex");
          $("#pop_up_text").html(text);

      })

  }

  $("#connect").click(connect_handler);
  connect_handler();

  const canvas = document.getElementById("canvas");


  activeBall = null;




  function drawHexagon(x, y, col, row, playable = false) {
      hexagon = document.createElement("div");
      hexagon.classList.add("hexagon");
      hexagon.style.left = x + "px";
      hexagon.style.top = y + "px";
      hexagon.id = "hexa_" + row + "_" + col;
      canvas.appendChild(hexagon);
      if (playable) hexagon.classList.add("playable");
      hexagon.addEventListener('click', onBallClick);

  }

  function addBall(col, row, color) {
      hexagon = document.getElementById("hexa_" + row + "_" + col);
      ball = document.createElement("div");
      ball.classList.add("ball");
      ball.classList.add(color);
      ball.style.backgroundColor = color;
      ball.id = "ball_" + row + "_" + col;
      hexagon.appendChild(ball);

  }

  function score_balls(color, number) {
      score = document.getElementById(color + "_score");
      score.innerHTML = "";
      for (i = 0; i < number; i++) {
          ball = document.createElement("div");
          ball.classList.add("ball");
          ball.classList.add(color);
          ball.style.backgroundColor = color;
          score.appendChild(ball);
      }
  }

  function onBallClick(event) {
      if (activeBall == null && event.currentTarget.classList.contains("playable")) {
          activeBall = event.currentTarget;
          activeBall.classList.add("active");
          x = parseInt(event.currentTarget.id.split("_")[1]);
          y = parseInt(event.currentTarget.id.split("_")[2]);
          lightNeighbours(x, y);
      } else {
          if (event.currentTarget.classList.contains("selected")) {
              secondLoc = event.currentTarget
              unlightNeighbours();
              secondLoc.classList.add("active");
              setTimeout(function() {
                  activeBall.classList.remove("active");
                  secondLoc.classList.remove("active");
                  loc = activeBall.id.split("_");
                  type = document.getElementById("ball_" + loc[1] + "_" + loc[2]).classList.contains("white") ? "W" : "B";
                  socket.emit("interact", JSON.stringify({
                      "from": getLocation(activeBall),
                      "to": getLocation(secondLoc),
                      "type": type
                  }));
                  activeBall = null;
                  secondLoc = null;
              }, 1000);
          } else {

              activeBall.classList.remove("active");
              activeBall = null
              unlightNeighbours();
          }
      }



  }

  function lightNeighbours(x, y) {
      neighbours = [
          [x - 1, y],
          [x + 1, y],
          [x, y - 1],
          [x, y + 1]
      ];
      if (x % 2 == 0) {
          neighbours.push([x + 1, y - 1]);
          neighbours.push([x - 1, y - 1]);
      } else {
          neighbours.push([x - 1, y + 1]);
          neighbours.push([x + 1, y + 1]);
      }
      for (i = 0; i < neighbours.length; i++) {
          neighbour = document.getElementById("hexa_" + neighbours[i][0] + "_" + neighbours[i][1]);
          if (neighbour) {
              neighbour.classList.add("selected");
          }
      }
  }

  function unlightNeighbours() {
      neighbours = document.getElementsByClassName("selected");
      neighbours = Array.from(neighbours);
      neighbours.forEach((element) => {
          element.classList.remove('selected');
      });
  }

  function getLocation(ball) {
      id = ball.id.split("_");
      current_location = [parseInt(id[1]), parseInt(id[2])];
      conversion = {
          "0_6": [0, 4],
          "0_5": [1, 3],
          "0_4": [2, 2],
          "0_3": [3, 1],
          "0_2": [4, 0],
          "1_6": [1, 5],
          "1_5": [2, 4],
          "1_4": [3, 3],
          "1_3": [4, 2],
          "1_2": [5, 1],
          "1_1": [6, 0],
          "2_7": [2, 6],
          "2_6": [3, 5],
          "2_5": [4, 4],
          "2_4": [5, 3],
          "2_3": [6, 2],
          "2_2": [7, 1],
          "2_1": [8, 0],
          "3_7": [3, 7],
          "3_6": [4, 6],
          "3_5": [5, 5],
          "3_4": [6, 4],
          "3_3": [7, 3],
          "3_2": [8, 2],
          "3_1": [9, 1],
          "3_0": [10, 0],
          "4_8": [4, 8],
          "4_7": [5, 7],
          "4_6": [6, 6],
          "4_5": [7, 5],
          "4_4": [8, 4],
          "4_3": [9, 3],
          "4_2": [10, 2],
          "4_1": [11, 1],
          "4_0": [12, 0],
          "5_7": [6, 8],
          "5_6": [7, 7],
          "5_5": [8, 6],
          "5_4": [9, 5],
          "5_3": [10, 4],
          "5_2": [11, 3],
          "5_1": [12, 2],
          "5_0": [13, 1],
          "6_7": [8, 8],
          "6_6": [9, 7],
          "6_5": [10, 6],
          "6_4": [11, 5],
          "6_3": [12, 4],
          "6_2": [13, 3],
          "6_1": [14, 2],
          "7_6": [10, 8],
          "7_5": [11, 7],
          "7_4": [12, 6],
          "7_3": [13, 5],
          "7_2": [14, 4],
          "7_1": [15, 3],
          "8_6": [12, 8],
          "8_5": [13, 7],
          "8_4": [14, 6],
          "8_3": [15, 5],
          "8_2": [16, 4]
      }

      return conversion[current_location[0] + "_" + current_location[1]];
  }

  function drawGrid(gridData) {

      scores = gridData["scores"];
      next_player = gridData["next_player"];
      current_step = gridData["current_step"];
      gridData = gridData["gridData"];

      $("#steps").html(current_step);


      canvas.innerHTML = ""


      // Calculate the available width and height for the grid
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;

      const gridSize = 9;
      const cellSize = Math.min(canvasWidth, canvasHeight) / gridSize;

      // Calculate the horizontal and vertical offset to center the grid
      const gridWidth = gridSize * cellSize;
      const gridHeight = gridSize * cellSize;
      const offsetX = (canvasWidth - gridWidth) / 2 + 125;
      const offsetY = (canvasHeight - gridHeight) / 2;

      // Define the grid data with the initial Abalone state


      // Define hexagon radius and spacing
      const hexagonWidth = 100;
      const hexagonHeight = 115;

      // Calculate the vertical offset to move the grid downwards
      //const marginTop = hexagonRadius * 4;



      for (let row = 0; row < gridData.length; row++) {
          for (let col = 0; col < gridData[row].length; col++) {
              let x = col * (hexagonWidth - 2);
              let y = row * (hexagonHeight - 32);

              // Adjust x coordinate for odd rows
              if (row % 2 !== 0) {
                  x += hexagonWidth / 2;
              }


              const value = gridData[row][col];
              // 1 = black, 2 = white, 3 = empty
              // Draw hexagon border for values different from 0

              if (value !== 0) {
                  if (value === 1) {
                      drawHexagon(x, y, col, row, next_player && next_player.player_type == "interactive" && next_player.piece_type == "B");
                  } else if (value === 2) {
                      drawHexagon(x, y, col, row, next_player && next_player.player_type == "interactive" && next_player.piece_type == "W");
                  } else {
                      drawHexagon(x, y, col, row);
                  }

                  // Draw circle inside the hexagon
                  if (value !== 3) {
                      color = value === 1 ? "black" : "white";
                      addBall(col, row, color);
                  }
              }
          }
      }


      const blackCircles = scores['B'] || 0;
      score_balls("black", blackCircles);
      const whiteCircles = scores['W'] || 0;
      score_balls("white", whiteCircles);



  }

  $("#next").click(function() {
      if (index < steps.length - 1) {
          index++;
          drawGrid(steps[index]);
      }
  });

  $("#previous").click(function() {
      if (index > 0) {
          index--;
          drawGrid(steps[index]);
      }
  });

  function convertToGridData(board) {
      scores = board["scores"]
      next_player = board["next_player"]
      players = board["players"]
      current_step = board["current_step"]
      board = board["board"]

      conversion = {
          '0_4': [0, 6],
          '1_3': [0, 5],
          '2_2': [0, 4],
          '3_1': [0, 3],
          '4_0': [0, 2],
          '1_5': [1, 6],
          '2_4': [1, 5],
          '3_3': [1, 4],
          '4_2': [1, 3],
          '5_1': [1, 2],
          '6_0': [1, 1],
          '2_6': [2, 7],
          '3_5': [2, 6],
          '4_4': [2, 5],
          '5_3': [2, 4],
          '6_2': [2, 3],
          '7_1': [2, 2],
          '8_0': [2, 1],
          '3_7': [3, 7],
          '4_6': [3, 6],
          '5_5': [3, 5],
          '6_4': [3, 4],
          '7_3': [3, 3],
          '8_2': [3, 2],
          '9_1': [3, 1],
          '10_0': [3, 0],
          '4_8': [4, 8],
          '5_7': [4, 7],
          '6_6': [4, 6],
          '7_5': [4, 5],
          '8_4': [4, 4],
          '9_3': [4, 3],
          '10_2': [4, 2],
          '11_1': [4, 1],
          '12_0': [4, 0],
          '6_8': [5, 7],
          '7_7': [5, 6],
          '8_6': [5, 5],
          '9_5': [5, 4],
          '10_4': [5, 3],
          '11_3': [5, 2],
          '12_2': [5, 1],
          '13_1': [5, 0],
          '8_8': [6, 7],
          '9_7': [6, 6],
          '10_6': [6, 5],
          '11_5': [6, 4],
          '12_4': [6, 3],
          '13_3': [6, 2],
          '14_2': [6, 1],
          '10_8': [7, 6],
          '11_7': [7, 5],
          '12_6': [7, 4],
          '13_5': [7, 3],
          '14_4': [7, 2],
          '15_3': [7, 1],
          '12_8': [8, 6],
          '13_7': [8, 5],
          '14_6': [8, 4],
          '15_5': [8, 3],
          '16_4': [8, 2]
      }


      score_w = 0;
      score_b = 0;

      for (i = 0; i < players.length; i++) {

          if (typeof players[i] === 'string' || players[i] instanceof String) {
              players[i] = {
                  "id": parseInt(players[i])
              }
              pt = Object.entries(board).filter(e => e[1]["owner_id"] == players[i].id)[0][1].piece_type
              players[i]["piece_type"] = pt
          }

          if (players[i].piece_type == "W") score_w = -scores[players[i].id];
          if (players[i].piece_type == "B") score_b = -scores[players[i].id];
      }
      real_scores = {
          "W": score_w,
          "B": score_b
      }

      const gridData = [
          [0, 0, 3, 3, 3, 3, 3, 0, 0],
          [0, 3, 3, 3, 3, 3, 3, 0, 0],
          [0, 3, 3, 3, 3, 3, 3, 3, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 0],
          [3, 3, 3, 3, 3, 3, 3, 3, 3],
          [3, 3, 3, 3, 3, 3, 3, 3, 0],
          [0, 3, 3, 3, 3, 3, 3, 3, 0],
          [0, 3, 3, 3, 3, 3, 3, 0, 0],
          [0, 0, 3, 3, 3, 3, 3, 0, 0],
      ];
      for (let key in board) {
          new_key = key.substring(1, key.length - 1).replace(/, /g, "_");
          obj = board[key];
          let new_coord = conversion[new_key];
          gridData[new_coord[0]][new_coord[1]] = obj["piece_type"]
      }

      for (i = 0; i < 9; i++) {
          for (j = 0; j < 9; j++) {
              if (gridData[i][j] == "W") {
                  gridData[i][j] = 2;
              } else if (gridData[i][j] == "B") {
                  gridData[i][j] = 1;
              }
          }
      }
      return {
          "gridData": gridData,
          "scores": real_scores,
          "next_player": next_player,
          "players": players,
          "current_step": current_step
      };
  }

  function resizeCanvas() {

      const defaultGridData = {
          "gridData": [
              [0, 0, 3, 3, 3, 3, 3, 0, 0],
              [0, 3, 3, 3, 3, 3, 3, 0, 0],
              [0, 3, 3, 3, 3, 3, 3, 3, 0],
              [3, 3, 3, 3, 3, 3, 3, 3, 0],
              [3, 3, 3, 3, 3, 3, 3, 3, 3],
              [3, 3, 3, 3, 3, 3, 3, 3, 0],
              [0, 3, 3, 3, 3, 3, 3, 3, 0],
              [0, 3, 3, 3, 3, 3, 3, 0, 0],
              [0, 0, 3, 3, 3, 3, 3, 0, 0],
          ],
          "scores": {
              "W": 0,
              "B": 0
          }
      };

      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      drawGrid(steps[index] || defaultGridData);
  }

  // Resize canvas when the window is resized
  window.addEventListener("resize", resizeCanvas);

  // Initial canvas setup
  resizeCanvas();

  $("#close_pop_up").click(function() {
      $("#pop_up_container").css("display", "none");

  });
});