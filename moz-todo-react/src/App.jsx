import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <body>
      <div className="Inputs">
        
        <label>Day:</label>
        <input name="Day" type="text"></input>
        <label>Hour:</label>
        <input name="Hour" type="number"></input>
        <list>
          <input name="Repost Input" value="Repost true" type="radio"></input>
          <label htmlFor="Repost true">Repost</label>
          <input name="Repost Input" value= "Repost false" type="radio"></input>
          <label htmlFor="Repost false">Not a repost</label>

        </list>
      </div>
      <div>
        <button>Submit</button>
      </div>
     </body>
  ) 
}

export default App
