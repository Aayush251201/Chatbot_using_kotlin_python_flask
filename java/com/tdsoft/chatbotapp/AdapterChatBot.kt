package com.tdsoft.chatbotapp

import android.graphics.Color
import android.view.Gravity
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import kotlinx.android.synthetic.main.listitem_chat.view.*

class AdapterChatBot : RecyclerView.Adapter<AdapterChatBot.MyViewHolder>() {
    private val list = ArrayList<ChatModel>()

    inner class MyViewHolder(parent: ViewGroup) : RecyclerView.ViewHolder(
        LayoutInflater.from(parent.context).inflate(R.layout.listitem_chat, parent, false)
    ) {
        fun bind(chat: ChatModel) = with(itemView) {
            if(!chat.isBot) {
                txtChat.setBackgroundColor(Color.BLUE)
                txtChat.setTextColor(Color.WHITE)
                txtChat.text = chat.chat
                vw.gravity=Gravity.RIGHT
            }else{
                txtChat.setBackgroundColor(Color.LTGRAY)
                txtChat.setTextColor(Color.WHITE)
                txtChat.text = chat.chat
                vw.gravity=Gravity.LEFT

            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) = MyViewHolder(parent)

    override fun onBindViewHolder(holder: MyViewHolder, position: Int) {
        holder.bind(list[position])
    }

    override fun getItemCount() = list.size

    fun addChatToList(chat: ChatModel) {
        list.add(chat)
        notifyDataSetChanged()
    }

}